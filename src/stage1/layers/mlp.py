# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/mlp.py


from torch import Tensor, nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        return self.fc2(x)
