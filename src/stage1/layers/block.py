# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

from torch import Tensor, nn

from .attention import Attention
from .layer_scale import LayerScale
from .mlp import Mlp


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        rope,
        mlp_ratio: float = 4.0,
        init_values: float = 0.01,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, rope=rope)
        self.ls1 = LayerScale(dim, init_values=init_values)

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
        )
        self.ls2 = LayerScale(dim, init_values=init_values)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), positions))
        return x + self.ls2(self.mlp(self.norm2(x)))
