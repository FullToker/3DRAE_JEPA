# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float | Tensor = 1e-5,
    ) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma
