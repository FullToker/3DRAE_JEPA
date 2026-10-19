# Copyright 2022 Facebook AI and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..layers import Block, PatchEmbed, PositionGetter, RotaryPositionEmbedding2D
from .dpt_head import DPTHead


@dataclass
class PointDecoderOutput:
    pts3d: torch.Tensor
    pts3d_conf: torch.Tensor


class LatentPointDecoder(nn.Module):
    def __init__(
        self,
        hidden_size_input=2048,
        hidden_size=768,
        patch_size=14,
        depth=16,
        num_heads=16,
        mlp_ratio=4,
        rope_freq=100,
        init_values=0.01,
        intermediate_layer_idx=(3, 7, 11, 15),
    ):
        super().__init__()
        intermediate_layer_idx = tuple(intermediate_layer_idx)
        if len(intermediate_layer_idx) != 4 or tuple(
            sorted(set(intermediate_layer_idx))
        ) != intermediate_layer_idx:
            raise ValueError(
                "Point decoder requires four unique, increasing intermediate layers"
            )
        if intermediate_layer_idx[0] < 0 or intermediate_layer_idx[-1] >= depth:
            raise ValueError("intermediate_layer_idx is outside the decoder depth")
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.intermediate_layer_idx = intermediate_layer_idx
        self.decoder_embed = nn.Linear(hidden_size_input, hidden_size)
        if rope_freq <= 0:
            raise ValueError("rope_freq must be positive")
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq)
        self.position_getter = PositionGetter()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=hidden_size,
                    num_heads=num_heads,
                    rope=self.rope,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                )
                for _ in range(depth)
            ]
        )
        self.ray_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=6,
            hidden_size=hidden_size,
        )
        self.decoder_dpt = DPTHead(
            dim_in=hidden_size,
            patch_size=patch_size,
        )

    def _prepare_rope(self, batch_size, views, height, width, patches, latents, device):
        pos_views = self.position_getter(
            batch_size * views,
            height // self.patch_size,
            width // self.patch_size,
            device=device,
        ).view(batch_size * views, patches, 2)
        pos_latents = torch.zeros(
            batch_size, latents, 2, dtype=pos_views.dtype, device=device
        )
        pos_latents = pos_latents.unsqueeze(1).repeat(1, views, 1, 1)
        return pos_views, pos_latents.view(batch_size * views, latents, 2)

    def forward(self, latent_tokens, ray_maps):
        batch_size, views, height, width, _ = ray_maps.shape
        _, latent_count, _ = latent_tokens.shape
        ray_tokens = self.ray_embed(
            ray_maps.view(batch_size * views, height, width, 6).permute(0, 3, 1, 2)
        )
        patch_count = ray_tokens.shape[1]
        pos_views, pos_latents = self._prepare_rope(
            batch_size,
            views,
            height,
            width,
            patch_count,
            latent_count,
            ray_tokens.device,
        )
        latent_tokens = self.decoder_embed(latent_tokens)
        latent_tokens = latent_tokens.unsqueeze(1).repeat(1, views, 1, 1)
        latent_tokens = latent_tokens.view(batch_size * views, latent_count, -1)
        tokens = torch.cat([latent_tokens, ray_tokens], dim=1)
        positions = torch.cat([pos_latents, pos_views], dim=1)

        aggregated = []
        selected_layers = set(self.intermediate_layer_idx)
        for layer_index, block in enumerate(self.blocks):
            if self.training:
                tokens = checkpoint(block, tokens, positions, use_reentrant=False)
            else:
                tokens = block(tokens, positions)
            if layer_index in selected_layers:
                aggregated.append(
                    tokens[:, latent_count:].view(
                        batch_size, views, -1, self.hidden_size
                    )
                )

        with torch.autocast(device_type=latent_tokens.device.type, enabled=False):
            points, confidence = self.decoder_dpt(
                aggregated,
                input_shape=(batch_size, views, height, width),
            )
        return PointDecoderOutput(pts3d=points, pts3d_conf=confidence)
