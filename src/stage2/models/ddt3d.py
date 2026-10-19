import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .model_utils import (
    GaussianFourierEmbedding,
    LabelEmbedder,
    NormAttention,
    RMSNorm,
    SwiGLUFFN,
    get_2d_sincos_pos_embed,
)


def ddt_modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    if shift.shape[1] not in (1, x.shape[1]):
        raise ValueError(
            f"Condition length must be 1 or {x.shape[1]}, got {shift.shape[1]}"
        )
    return x * (1 + scale) + shift


def ddt_gate(x: torch.Tensor, gate: torch.Tensor):
    if gate.shape[1] not in (1, x.shape[1]):
        raise ValueError(
            f"Condition length must be 1 or {x.shape[1]}, got {gate.shape[1]}"
        )
    return x * gate


class LightningDDTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.attn = NormAttention(hidden_size, num_heads=num_heads)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_hidden_dim))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        modulation = self.adaLN_modulation(c).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
        attention = self.attn(ddt_modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + ddt_gate(attention, gate_msa)
        mlp = self.mlp(ddt_modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + ddt_gate(mlp, gate_mlp)
        return x


class DDTFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(ddt_modulate(self.norm_final(x), shift, scale))


class DiTwDDTHead(nn.Module):
    def __init__(
        self,
        latent_size: int = 1024,
        in_channels: int = 768,
        hidden_size=(1152, 2048),
        depth=(28, 2),
        num_heads: list[int] | int = (16, 16),
        mlp_ratio=4.0,
        num_classes=1000,
    ):
        super().__init__()
        if len(hidden_size) != 2 or len(depth) != 2:
            raise ValueError("hidden_size and depth must each have two entries")
        self.encoder_hidden_size = hidden_size[0]
        self.decoder_hidden_size = hidden_size[1]
        num_heads = (
            [num_heads, num_heads]
            if isinstance(num_heads, int)
            else list(num_heads)
        )
        if len(num_heads) != 2:
            raise ValueError("num_heads must have two entries")
        self.num_encoder_blocks = depth[0]
        num_blocks = depth[0] + depth[1]
        self.num_classes = num_classes
        self.latent_size = latent_size
        self.x_embedder = nn.Linear(
            2 * in_channels, self.decoder_hidden_size, bias=True
        )
        self.s_embedder = nn.Linear(
            2 * in_channels, self.encoder_hidden_size, bias=True
        )
        self.s_projector = (
            nn.Linear(self.encoder_hidden_size, self.decoder_hidden_size)
            if self.encoder_hidden_size != self.decoder_hidden_size
            else nn.Identity()
        )
        self.t_embedder = GaussianFourierEmbedding(self.encoder_hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, self.encoder_hidden_size)
        self.final_layer = DDTFinalLayer(
            self.decoder_hidden_size,
            in_channels,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, latent_size, self.encoder_hidden_size),
            requires_grad=False,
        )
        enc_num_heads = num_heads[0]
        dec_num_heads = num_heads[1]
        self.blocks = nn.ModuleList(
            [
                LightningDDTBlock(
                    (
                        self.encoder_hidden_size
                        if index < self.num_encoder_blocks
                        else self.decoder_hidden_size
                    ),
                    (
                        enc_num_heads
                        if index < self.num_encoder_blocks
                        else dec_num_heads
                    ),
                    mlp_ratio=mlp_ratio,
                )
                for index in range(num_blocks)
            ]
        )
        self.initialize_weights()

    def initialize_weights(self):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        w = self.s_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], (self.latent_size, 1)
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, ci=None):
        t = self.t_embedder(t)
        labels = torch.full((x.shape[0],), self.num_classes, device=x.device)
        y = self.y_embedder(labels)
        c = nn.functional.silu(t + y)
        if ci is None:
            raise ValueError("ci condition is required")
        x = torch.cat([x, ci], dim=-1)
        s = self.s_embedder(x)
        s = s + self.pos_embed
        for block in self.blocks[: self.num_encoder_blocks]:
            if self.training:
                s = checkpoint(block, s, c, use_reentrant=False)
            else:
                s = block(s, c)
        t = t.unsqueeze(1).expand(-1, s.shape[1], -1)
        s = nn.functional.silu(t + s)
        s = self.s_projector(s)
        x = self.x_embedder(x)
        for block in self.blocks[self.num_encoder_blocks :]:
            if self.training:
                x = checkpoint(block, x, s, use_reentrant=False)
            else:
                x = block(x, s)
        if self.training:
            x = checkpoint(self.final_layer, x, s, use_reentrant=False)
        else:
            x = self.final_layer(x, s)
        return x

    def forward_with_cfg(
        self,
        x,
        t,
        ci=None,
        cfg_scale=1.0,
        cfg_interval=(0, 1),
    ):
        if len(x) % 2:
            raise ValueError("Classifier-free guidance requires an even batch size")
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, ci)
        cond_eps, uncond_eps = torch.chunk(model_out, 2, dim=0)
        guid_t_min, guid_t_max = cfg_interval
        if guid_t_min >= guid_t_max:
            raise ValueError("cfg_interval must satisfy min < max")
        t = t[: len(t) // 2]
        half_eps = torch.where(
            ((t >= guid_t_min) & (t <= guid_t_max)).view(
                -1, *[1] * (len(cond_eps.shape) - 1)
            ),
            uncond_eps + cfg_scale * (cond_eps - uncond_eps),
            cond_eps,
        )
        return torch.cat([half_eps, half_eps], dim=0)
