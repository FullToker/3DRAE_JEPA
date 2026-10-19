# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import torch.nn.functional as F
from torch import Tensor, nn


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        rope,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.rope = rope

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        batch_size, token_count, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(
                batch_size,
                token_count,
                3,
                self.num_heads,
                channels // self.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)
        q = self.rope(q, positions)
        k = self.rope(k, positions)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(batch_size, token_count, channels)
        return self.proj(x)
