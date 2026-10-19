import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    if embed_dim % 2:
        raise ValueError("Embedding dimension must be even")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2:
        raise ValueError("Embedding dimension must be even")
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_size_h, grid_size_w = grid_size
    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0)
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        output = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return output.type_as(x) * self.weight


class NormAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, token_count, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v)
        x = x.transpose(1, 2).reshape(batch_size, token_count, channels)
        return self.proj(x)


class GaussianFourierEmbedding(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = nn.Parameter(
            torch.normal(0, 1.0, (256,)), requires_grad=False
        )
        self.mlp = nn.Sequential(
            nn.Linear(512, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t):
        t = t[:, None] * self.W.detach()[None, :] * 2 * torch.pi
        return self.mlp(torch.cat([torch.sin(t), torch.cos(t)], dim=-1))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)

    def forward(self, labels):
        return self.embedding_table(labels)
