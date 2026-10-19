import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..layers import Block, PatchEmbed, PositionGetter, RotaryPositionEmbedding2D


class LatentFuseNeck(nn.Module):
    def __init__(
        self,
        latent_token_num=1024,
        hidden_size_input: int = 2048,
        hidden_size: int = 768,
        patch_size: int = 14,
        depth=12,
        num_heads=16,
        mlp_ratio=4,
        rope_freq=100,
        init_values=0.01,
        normalize: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.project = nn.Linear(hidden_size_input, hidden_size, bias=True)
        self.mv_norm = nn.LayerNorm(hidden_size)

        self.latent_token_num = latent_token_num
        self.latent_tokens = nn.Parameter(torch.randn(1, latent_token_num, hidden_size))
        nn.init.trunc_normal_(self.latent_tokens, std=0.02)

        self.pose_img_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=7,
            hidden_size=hidden_size,
        )

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
        self.latent_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=not normalize
        )

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
        )
        pos_views = pos_views.view(batch_size, views * patches, 2)
        pos_latents = torch.zeros(
            batch_size,
            latents,
            2,
            dtype=pos_views.dtype,
            device=device,
        )
        return torch.cat([pos_latents, pos_views], dim=1)

    def forward(self, features_visible, visible_ids, ray_maps):
        batch_size, views, height, width, _ = ray_maps.shape
        if features_visible is not None:
            _, _, patch_count, channels = features_visible.shape
            features_visible = features_visible.view(
                batch_size, -1, patch_count, channels
            )
            mv_tokens_visible = self.mv_norm(self.project(features_visible))
            batch_ids = torch.arange(
                batch_size, device=mv_tokens_visible.device
            ).view(-1, 1)
            if visible_ids is not None:
                mv_tokens = torch.zeros(
                    batch_size,
                    views,
                    patch_count,
                    self.hidden_size,
                    device=mv_tokens_visible.device,
                    dtype=mv_tokens_visible.dtype,
                )
                mv_tokens[batch_ids, visible_ids] = mv_tokens_visible
                mv_masks = torch.zeros(
                    batch_size,
                    views,
                    1,
                    height,
                    width,
                    device=mv_tokens.device,
                    dtype=mv_tokens.dtype,
                )
                mv_masks[batch_ids, visible_ids] = 1.0
            else:
                mv_tokens = mv_tokens_visible
                mv_masks = torch.ones(
                    batch_size,
                    views,
                    1,
                    height,
                    width,
                    device=mv_tokens.device,
                    dtype=mv_tokens.dtype,
                )
        else:
            mv_tokens = None
            mv_masks = torch.zeros(
                batch_size,
                views,
                1,
                height,
                width,
                device=ray_maps.device,
                dtype=ray_maps.dtype,
            )

        ray_maps = ray_maps.permute(0, 1, 4, 2, 3)
        ray_mask_maps = torch.cat([ray_maps, mv_masks], dim=2).view(
            batch_size * views, 7, height, width
        )
        pose_tokens = self.pose_img_embed(ray_mask_maps).reshape(
            batch_size, views, -1, self.hidden_size
        )
        patch_count = pose_tokens.shape[2]
        mv_tokens = pose_tokens if mv_tokens is None else mv_tokens + pose_tokens
        mv_tokens = mv_tokens.reshape(
            batch_size, views * patch_count, self.hidden_size
        )

        latent_tokens = self.latent_tokens.repeat(batch_size, 1, 1)
        latent_count = latent_tokens.shape[1]
        positions = self._prepare_rope(
            batch_size,
            views,
            height,
            width,
            patch_count,
            latent_count,
            latent_tokens.device,
        )
        tokens = torch.cat([latent_tokens, mv_tokens], dim=1)

        for block in self.blocks:
            if self.training:
                tokens = checkpoint(
                    block, tokens, positions, use_reentrant=False
                )
            else:
                tokens = block(tokens, positions)
        return self.latent_norm(tokens[:, : self.latent_token_num])
