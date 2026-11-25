"""dino.txt's image tower as the VL-JEPA X-encoder, in place of 3DRAE.

A baseline that asks what language-aligned patch features alone give: there is
no 3D fusion across views here, the only geometry is each token's camera ray.

Input  (as jepa3d.x_encoder.ThreeDRAEXEncoder):
    images : (B, V, 3, H, W) float [0, 1]      cams : (B, V, 25) [K | normalized w2c]
Output :
    tokens : (B, V * P + V, dim)
        P tokens per view: dino.txt's text-aligned patch features (frozen ViT-L/14),
          averaged over pool x pool patches (224 x 308, pool 2: 8 x 11 = 88 per view,
          704 for 8 views), projected, plus an embedding of the Plucker ray through
          the cell centre (first-camera frame) and of the view index;
        V view tokens as in ThreeDRAEXEncoder (mean feature + camera pose + view index).
        Tokens of padded views are a learned pad token.
Trainable: the projections and embeddings. The dino.txt model is frozen.
"""

import torch
import torch.nn.functional as F
from torch import nn

from jepa3d.x_encoder import unpack_cams, view_counts

PATCH = 14
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class DinoTxtXEncoder(nn.Module):
    def __init__(self, dim: int = 768, pool: int = 2, max_views: int = 32):
        super().__init__()
        from jepa3d.tools.check_teacher_grounding import load_dinotxt
        self.teacher, self.tokenizer = load_dinotxt("cpu")
        self._text = {}   # name -> text vector (cpu), filled on demand
        self.last = None  # set by forward()
        self.teacher.requires_grad_(False)
        self.dim, self.pool, self.max_views = dim, pool, max_views
        with torch.no_grad():
            _, patches = self.teacher.get_visual_class_and_patch_tokens(torch.zeros(1, 3, 2 * PATCH, 2 * PATCH))
        feat_dim = patches.shape[-1]
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

        self.feat_proj = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, dim))
        self.ray_mlp = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, dim))
        self.view_feat_proj = nn.Sequential(nn.LayerNorm(feat_dim), nn.Linear(feat_dim, dim))
        self.view_pose_mlp = nn.Sequential(nn.Linear(16, dim), nn.GELU(), nn.Linear(dim, dim))
        self.view_index_embed = nn.Embedding(max_views, dim)
        self.token_type_embed = nn.Embedding(2, dim)   # 0 = patch, 1 = view
        self.pad_token = nn.Parameter(torch.zeros(dim))
        for emb in (self.view_index_embed, self.token_type_embed):
            nn.init.normal_(emb.weight, std=0.02)

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def text_vectors(self, names) -> torch.Tensor:
        """Object names -> (N, C) frozen text vectors, the half that is aligned with patch features."""
        new = sorted({n for n in names if n not in self._text})
        device = self.mean.device
        for start in range(0, len(new), 64):
            chunk = new[start:start + 64]
            tokens = self.tokenizer.tokenize([f"a photo of a {n}" for n in chunk]).to(device)
            features = self.teacher.encode_text(tokens).float()
            for name, vector in zip(chunk, features[:, features.shape[-1] // 2:].cpu()):
                self._text[name] = vector
        return torch.stack([self._text[n] for n in names]).to(device)

    @torch.no_grad()
    def _features(self, images):
        """(B, V, 3, H, W) -> (B, V, h, w, C) pooled text-aligned patch features."""
        B, V, _, height, width = images.shape
        _, patches = self.teacher.get_visual_class_and_patch_tokens((images.flatten(0, 1) - self.mean) / self.std)
        grid = patches.float().unflatten(1, (height // PATCH, width // PATCH)).permute(0, 3, 1, 2)
        grid = F.avg_pool2d(grid, self.pool)
        return grid.permute(0, 2, 3, 1).unflatten(0, (B, V))

    @torch.no_grad()
    def _rays(self, K, w2c, h, w):
        """K (B, V, 3, 3), w2c (B, V, 4, 4) -> Plucker rays (B, V, h * w, 6) through the cell centres."""
        cell = PATCH * self.pool
        v, u = torch.meshgrid((torch.arange(h, device=K.device) + 0.5) * cell,
                              (torch.arange(w, device=K.device) + 0.5) * cell, indexing="ij")
        pixels = torch.stack([u, v, torch.ones_like(u)], dim=-1).reshape(1, 1, -1, 3, 1).float()
        direction = F.normalize(torch.linalg.solve(K[:, :, None], pixels).squeeze(-1), dim=-1)
        rotation_t = w2c[..., :3, :3].transpose(-1, -2)
        direction = (rotation_t[:, :, None] @ direction.unsqueeze(-1)).squeeze(-1)
        centre = -(rotation_t @ w2c[..., :3, 3:]).squeeze(-1)[:, :, None].expand_as(direction)
        return torch.cat([direction, torch.cross(centre, direction, dim=-1)], dim=-1)

    def forward(self, images: torch.Tensor, cams: torch.Tensor) -> torch.Tensor:
        B, V, _, height, width = images.shape
        if V > self.max_views:
            raise ValueError(f"{V} views exceed max_views={self.max_views}")
        counts = view_counts(cams)
        valid = torch.arange(V, device=cams.device)[None] < counts[:, None]            # (B, V)
        K, w2c = (x.float() for x in unpack_cams(cams))
        K = torch.where(valid[..., None, None], K, torch.eye(3, device=K.device))       # padded views: K == 0

        feats = self._features(images)                                                  # (B, V, h, w, C)
        h, w = feats.shape[2:4]
        feats = feats.flatten(2, 3)                                                     # (B, V, P, C)
        index = self.view_index_embed.weight[:V]
        rays = self._rays(K, w2c, h, w)
        # for jepa3d.pointer: the frozen features, which views are real, the rays and the grid of this batch
        self.last = {"features": feats, "valid": valid, "rays": rays, "grid": (h, w)}
        patches = self.feat_proj(feats) + self.ray_mlp(rays) + index[None, :, None]
        patches = torch.where(valid[..., None, None], patches, self.pad_token.to(patches.dtype))
        patches = patches.flatten(1, 2) + self.token_type_embed.weight[0]

        c2w = torch.linalg.inv(w2c)
        scale = K.new_tensor([width, height])
        intr = torch.cat([K[..., [0, 1], [0, 1]] / scale, K[..., [0, 1], [2, 2]] / scale], dim=-1)
        pose = torch.cat([c2w[..., :3, 3], c2w[..., :3, :3].flatten(-2), intr], dim=-1)
        views = self.view_feat_proj(feats.mean(dim=2)) + self.view_pose_mlp(pose) + index
        views = torch.where(valid[..., None], views, self.pad_token.to(views.dtype))
        views = views + self.token_type_embed.weight[1]
        return torch.cat([patches, views.to(patches.dtype)], dim=1)
