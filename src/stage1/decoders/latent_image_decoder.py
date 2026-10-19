# Copyright 2022 Facebook AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..layers import Block, PatchEmbed, PositionGetter, RotaryPositionEmbedding2D


@dataclass
class DecoderOutput:
    logits: torch.Tensor


class LatentImageDecoder(nn.Module):
    def __init__(
        self,
        hidden_size_input: int = 2048,
        hidden_size: int = 768,
        patch_size: int = 14,
        depth=16,
        num_heads=16,
        mlp_ratio=4,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.decoder_embed = nn.Linear(hidden_size_input, hidden_size, bias=True)

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

        self.decoder_norm = nn.LayerNorm(hidden_size)
        self.decoder_pred = nn.Linear(
            hidden_size, patch_size**2 * 3, bias=True
        )
        self.decoder_act = nn.Sigmoid()

    def unpatchify(self, patchified_pixel_values, original_image_size):
        patch_size, num_channels = self.patch_size, 3
        original_height, original_width = original_image_size
        num_patches_h = original_height // patch_size
        num_patches_w = original_width // patch_size
        if num_patches_h * num_patches_w != patchified_pixel_values.shape[1]:
            raise ValueError(
                f"Expected {num_patches_h * num_patches_w} patches for "
                f"image size {original_image_size}, got "
                f"{patchified_pixel_values.shape[1]}"
            )

        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_h,
            num_patches_w,
            patch_size,
            patch_size,
            num_channels,
        )
        patchified_pixel_values = torch.einsum(
            "nhwpqc->nchpwq", patchified_pixel_values
        )
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_channels,
            num_patches_h * patch_size,
            num_patches_w * patch_size,
        )
        return pixel_values

    def _prepare_rope(
        self,
        batch_size,
        views,
        height,
        width,
        patches,
        latents,
        device,
    ):
        pos_views = self.position_getter(
            batch_size * views,
            height // self.patch_size,
            width // self.patch_size,
            device=device,
        ).view(batch_size * views, patches, 2)
        pos_latents = torch.zeros(
            batch_size,
            latents,
            2,
            dtype=pos_views.dtype,
            device=device,
        )
        pos_latents = pos_latents.unsqueeze(1).repeat(1, views, 1, 1)
        return pos_views, pos_latents.view(
            batch_size * views, latents, 2
        )

    def forward(self, latent_tokens, ray_maps):
        batch_size, views, height, width, _ = ray_maps.shape
        _, latent_count, _ = latent_tokens.shape
        ray_maps = ray_maps.view(batch_size * views, height, width, 6)
        ray_maps = ray_maps.permute(0, 3, 1, 2)
        ray_tokens = self.ray_embed(ray_maps)
        patch_count = ray_tokens.shape[1]

        pos_mv, pos_latent = self._prepare_rope(
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
        pos = torch.cat([pos_latent, pos_mv], dim=1)

        for block in self.blocks:
            if self.training:
                tokens = checkpoint(block, tokens, pos, use_reentrant=False)
            else:
                tokens = block(tokens, pos)
        image_tokens = tokens[:, latent_count:]
        image_tokens = self.decoder_norm(image_tokens)
        image_logits = self.decoder_pred(image_tokens)
        image_logits = self.decoder_act(image_logits)
        return DecoderOutput(logits=image_logits)
