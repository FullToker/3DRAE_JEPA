"""3DRAE as the VL-JEPA X-encoder.

Input  (packed by jepa3d.data.spar3d_collate):
    images : (B, V, 3, H, W) float [0, 1], zero-padded past each sample's view count
    cams   : (B, V, 25)      [K (9) | canonical normalized w2c (16)];
             padded slots have K == 0 and w2c == I
Output :
    tokens : (B, N + V, 768)
        N scene tokens from the frozen 3DRAE encoder (DINOv2 + LatentFuseNeck,
          normalized latent), shared across views -- no per-frame structure;
        V view tokens (trainable): per-view DINOv2 mean feature + camera pose +
          view index, so queries that name "the first/second image" can still
          tell views apart. Padded view slots become a learned pad token.
"""

from pathlib import Path

import torch
import yaml
from torch import nn

from jepa3d.paths import RAE_ROOT
from threedrae.feature_extractor import _resolve_config_paths
from utils import instantiate_from_config, plucker_rays_from_K_w2c

CAM_DIM = 25


def unpack_cams(cams: torch.Tensor):
    """(..., 25) -> K (..., 3, 3), w2c (..., 4, 4)."""
    return cams[..., :9].unflatten(-1, (3, 3)), cams[..., 9:].unflatten(-1, (4, 4))


def view_counts(cams: torch.Tensor) -> torch.Tensor:
    return (cams[..., 0] > 0).sum(dim=1)


class ThreeDRAEXEncoder(nn.Module):
    def __init__(
        self,
        feature_config: str = "configs/models/dino2_feature.yaml",
        max_views: int = 32,
    ):
        super().__init__()
        config_path = Path(feature_config)
        if not config_path.is_absolute():
            config_path = RAE_ROOT / config_path
        config = _resolve_config_paths(yaml.safe_load(config_path.read_text()))
        self.rae = instantiate_from_config(config)
        self.rae.requires_grad_(False)
        self.rae.eval()

        dim = self.rae.neck.hidden_size
        feat_dim = self.rae.encoder.hidden_size
        self.dim = dim
        self.num_latents = self.rae.neck.latent_token_num
        self.max_views = max_views

        self.view_feat_proj = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, dim))
        # camera centre (3) + rotation (9) + normalized intrinsics (4)
        self.view_pose_mlp = nn.Sequential(nn.Linear(16, dim), nn.GELU(), nn.Linear(dim, dim))
        self.view_index_embed = nn.Embedding(max_views, dim)
        self.token_type_embed = nn.Embedding(2, dim)   # 0 = scene latent, 1 = view
        self.pad_view_token = nn.Parameter(torch.zeros(dim))
        for emb in (self.view_index_embed, self.token_type_embed):
            nn.init.normal_(emb.weight, std=0.02)

    def train(self, mode: bool = True):
        super().train(mode)
        self.rae.eval()   # frozen: keep it in eval mode (no latent noising)
        return self

    @torch.no_grad()
    def _encode_group(self, images, K, w2c):
        """images (b, n, 3, H, W) with n real views -> latent (b, N, D), view feats (b, n, C)."""
        b, n, _, height, width = images.shape
        rays = torch.stack(
            [plucker_rays_from_K_w2c(k.float(), e.float(), height, width) for k, e in zip(K, w2c)]
        )
        rae = self.rae
        normalized = (images - rae.image_mean.unsqueeze(1)) / rae.image_std.unsqueeze(1)
        feats = rae.encoder(normalized.flatten(0, 1)).unflatten(0, (b, n))   # (b, n, P, C)
        latent = rae.neck(feats, None, rays)
        if rae.do_normalization:
            latent = rae._apply_norm(latent)
        return latent, feats.mean(dim=2)

    def _pose_features(self, K, w2c, height, width):
        c2w = torch.linalg.inv(w2c.float())
        scale = K.new_tensor([width, height])
        intr = torch.cat([K[..., [0, 1], [0, 1]] / scale, K[..., [0, 1], [2, 2]] / scale], dim=-1)
        return torch.cat([c2w[..., :3, 3], c2w[..., :3, :3].flatten(-2), intr], dim=-1)

    def forward(self, images: torch.Tensor, cams: torch.Tensor) -> torch.Tensor:
        B, V, _, height, width = images.shape
        if V > self.max_views:
            raise ValueError(f"{V} views exceed max_views={self.max_views}")
        counts = view_counts(cams)
        if (counts == 0).any():
            raise ValueError("every sample needs at least one view")
        K, w2c = unpack_cams(cams)

        latent = images.new_zeros(B, self.num_latents, self.dim, dtype=torch.float32)
        view_feat = images.new_zeros(B, V, self.rae.encoder.hidden_size, dtype=torch.float32)
        # 3DRAE needs the same view count within one forward: group samples by it.
        for n in counts.unique().tolist():
            idx = (counts == n).nonzero(as_tuple=True)[0]
            z, f = self._encode_group(images[idx, :n], K[idx, :n], w2c[idx, :n])
            latent[idx] = z.float()
            view_feat[idx, :n] = f.float()

        valid = torch.arange(V, device=cams.device)[None] < counts[:, None]   # (B, V)
        views = (
            self.view_feat_proj(view_feat)
            + self.view_pose_mlp(self._pose_features(K, w2c, height, width))
            + self.view_index_embed.weight[:V]
        )
        views = torch.where(valid[..., None], views, self.pad_view_token.to(views.dtype))
        latent = latent + self.token_type_embed.weight[0]
        views = views + self.token_type_embed.weight[1]
        return torch.cat([latent.to(views.dtype), views], dim=1)
