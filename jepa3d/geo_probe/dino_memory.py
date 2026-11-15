"""Probe memory from DINOv2 patch features, before the 3DRAE neck.

Same frozen DINOv2 and the same Plucker rays the neck receives, but no neck:
each view keeps its 16 x 22 patch tokens (224 x 308 input, patch 14), and each
patch gets its ray (mean over the patch's pixels) through a trainable linear
layer, plus a view index embedding. If the probe reads positions much better
from this than from the 1024 neck tokens, the neck loses them.
"""

import torch
from torch import nn

from jepa3d.x_encoder import encode_patches, unpack_cams
from utils import plucker_rays_from_K_w2c


def dino_patches(rae, images):
    """Frozen foundation-encoder patch features (DINOv2, or SigLIP2 with the SigLIP2
    3DRAE config): images (B, V, 3, H, W) -> (B, V, h, w, C), h = H // neck patch."""
    height, width = images.shape[-2:]
    patch = rae.neck.patch_size
    h, w = height // patch, width // patch
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        feats = encode_patches(rae, images).float()
    if feats.shape[2] != h * w:
        raise ValueError(f"expected {h * w} patches per view, got {feats.shape[2]}")
    return feats.unflatten(2, (h, w))


class PatchMemory(nn.Module):
    """Frozen patch features + each patch's Plucker ray + a view index embedding."""

    def __init__(self, feat_dim: int, max_views: int = 32):
        super().__init__()
        self.feat_dim = feat_dim
        self.norm = nn.LayerNorm(feat_dim)
        self.ray_proj = nn.Linear(6, feat_dim)
        self.view_embed = nn.Embedding(max_views, feat_dim)
        nn.init.normal_(self.view_embed.weight, std=0.02)

    def embed(self, feats, cams, height: int, width: int, patch: int):
        """feats (B, V, P, C) of a (height, width) input, cams (B, V, 25) -> tokens (B, V * P, C)."""
        B, V = feats.shape[:2]
        h, w = height // patch, width // patch
        with torch.no_grad():
            K, w2c = unpack_cams(cams)
            rays = torch.stack([plucker_rays_from_K_w2c(k.float(), e.float(), height, width)
                                for k, e in zip(K, w2c)])                       # (B, V, H, W, 6)
            rays = rays.reshape(B, V, h, patch, w, patch, 6).mean(dim=(3, 5)).flatten(2, 3)
        tokens = self.norm(feats) + self.ray_proj(rays) + self.view_embed.weight[:V, None]
        return tokens.flatten(1, 2)


class DinoPatchMemory(PatchMemory):
    def forward(self, rae, images, cams):
        """images (B, V, 3, H, W), cams (B, V, 25) -> tokens (B, V * P, C)."""
        feats = dino_patches(rae, images).flatten(2, 3)                    # (B, V, P, C)
        return self.embed(feats, cams, *images.shape[-2:], rae.neck.patch_size)
