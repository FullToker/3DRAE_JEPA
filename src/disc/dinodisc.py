import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import SpectralNorm


class MLPNoDrop(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class SelfAttentionNoDrop(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x):
        batch_size, token_count, channels = x.shape
        qkv = self.qkv(x).view(
            batch_size, token_count, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        attention = (q.mul(self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        output = attention @ v
        return self.proj(
            output.transpose(1, 2).reshape(batch_size, token_count, channels)
        )


class SABlockNoDrop(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio, norm_eps):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.attn = SelfAttentionNoDrop(embed_dim=embed_dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.mlp = MLPNoDrop(
            in_features=embed_dim,
            hidden_features=round(embed_dim * mlp_ratio),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.ratio = 1 / math.sqrt(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.fn(x).add(x)).mul_(self.ratio)


class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(
            self,
            name="weight",
            n_power_iterations=1,
            dim=0,
            eps=1e-12,
        )


class BatchNormLocal(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        mean = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))
        return x * self.weight[None, :, None] + self.bias[None, :, None]


def make_block(channels: int, kernel_size: int) -> nn.Module:
    return nn.Sequential(
        SpectralConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="circular",
        ),
        BatchNormLocal(channels),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


class DinoDisc(nn.Module):
    def __init__(
        self,
        device: torch.device,
        dino_ckpt_path: str,
        ks: int,
    ):
        super().__init__()
        state = torch.load(dino_ckpt_path, map_location="cpu")
        for key in sorted(state.keys()):
            if ".attn.qkv.bias" in key:
                bias = state[key]
                channels = bias.numel() // 3
                bias[channels : 2 * channels].zero_()

        dino = FrozenDINONoDrop()
        missing, unexpected = dino.load_state_dict(state, strict=False)
        missing = [
            name
            for name in missing
            if name not in {"x_scale", "x_shift"}
        ]
        if missing:
            raise RuntimeError(f"DINO checkpoint missing keys: {missing}")
        if unexpected:
            raise RuntimeError(f"DINO checkpoint has unexpected keys: {unexpected}")
        dino.eval()
        self.dino_proxy: tuple[FrozenDINONoDrop, ...] = (dino.to(device=device),)
        channels = self.dino_proxy[0].embed_dim
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    make_block(channels, kernel_size=1),
                    ResidualBlock(make_block(channels, kernel_size=ks)),
                    SpectralConv1d(channels, 1, kernel_size=1, padding=0),
                )
                for _ in range(5)
            ]
        )
        self.heads.requires_grad_(True)
        self.dino_proxy[0].requires_grad_(False)

    def forward(self, x_in_pm1: torch.Tensor) -> torch.Tensor:
        activations: list[torch.Tensor] = self.dino_proxy[0](x_in_pm1)
        batch = x_in_pm1.shape[0]
        outputs = []
        for head, act in zip(self.heads, activations):
            out = head(act).view(batch, -1)
            outputs.append(out)
        return torch.cat(outputs, dim=1)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class FrozenDINONoDrop(nn.Module):
    def __init__(self):
        super().__init__()
        embed_dim = 384
        patch_size = 8
        norm_eps = 1e-6
        self.embed_dim = 384
        self.img_size = 224
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
        )
        self.patch_nums = self.img_size // patch_size

        mean = torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.229, 0.224, 0.225))
        self.register_buffer("x_scale", (0.5 / std).reshape(1, 3, 1, 1))
        self.register_buffer("x_shift", ((0.5 - mean) / std).reshape(1, 3, 1, 1))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_nums * self.patch_nums + 1, embed_dim)
        )

        self.key_depths = {2, 5, 8, 11}
        self.blocks = nn.Sequential(
            *[
                SABlockNoDrop(
                    embed_dim=embed_dim,
                    num_heads=6,
                    mlp_ratio=4.0,
                    norm_eps=norm_eps,
                )
                for _ in range(12)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        x = F.interpolate(
            x,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        x = x * self.x_scale + self.x_shift
        batch_size = x.shape[0]

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed

        activations = []
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.key_depths:
                activations.append(x[:, 1:, :].transpose(1, 2))
        activations.insert(0, x[:, 1:, :].transpose(1, 2))
        return activations
