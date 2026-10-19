from torch import nn


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_channels: int, hidden_size: int):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)
