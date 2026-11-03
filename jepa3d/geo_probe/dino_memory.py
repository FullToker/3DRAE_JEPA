"""Probe memory from DINOv2 patch features, before the 3DRAE neck.

Same frozen DINOv2 and the same Plucker rays the neck receives, but no neck:
each view keeps its 16 x 22 patch tokens (224 x 308 input, patch 14), and each
patch gets its ray (mean over the patch's pixels) through a trainable linear
layer, plus a view index embedding. If the probe reads positions much better
from this than from the 1024 neck tokens, the neck loses them.
"""

import torch
from torch import nn

from jepa3d.x_encoder import unpack_cams
from utils import plucker_rays_from_K_w2c


class DinoPatchMemory(nn.Module):
    def __init__(self, feat_dim: int, max_views: int = 32):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)
        self.ray_proj = nn.Linear(6, feat_dim)
        self.view_embed = nn.Embedding(max_views, feat_dim)
        nn.init.normal_(self.view_embed.weight, std=0.02)

    def forward(self, rae, images, cams):
        """images (B, V, 3, H, W), cams (B, V, 25) -> tokens (B, V * P, C)."""
        B, V, _, height, width = images.shape
        patch = rae.encoder.patch_size
        h, w = height // patch, width // patch
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                normalized = (images - rae.image_mean.unsqueeze(1)) / rae.image_std.unsqueeze(1)
                feats = rae.encoder(normalized.flatten(0, 1)).float()
            if feats.shape[1] != h * w:
                raise ValueError(f"expected {h * w} patches per view, got {feats.shape[1]}")
            K, w2c = unpack_cams(cams)
            rays = torch.stack([plucker_rays_from_K_w2c(k.float(), e.float(), height, width)
                                for k, e in zip(K, w2c)])                       # (B, V, H, W, 6)
            rays = rays.reshape(B, V, h, patch, w, patch, 6).mean(dim=(3, 5)).flatten(2, 3)
        tokens = self.norm(feats.unflatten(0, (B, V))) + self.ray_proj(rays) + self.view_embed.weight[:V, None]
        return tokens.flatten(1, 2)
