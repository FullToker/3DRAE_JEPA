import torch
import torch.nn.functional as F
from torch import nn


class PositionGetter:
    def __init__(self):
        self.position_cache = {}

    def __call__(self, batch_size, height, width, device):
        key = (height, width, device)
        if key not in self.position_cache:
            y = torch.arange(height, device=device)
            x = torch.arange(width, device=device)
            self.position_cache[key] = torch.cartesian_prod(y, x)
        return (
            self.position_cache[key]
            .view(1, height * width, 2)
            .expand(batch_size, -1, -1)
            .clone()
        )


class RotaryPositionEmbedding2D(nn.Module):
    def __init__(self, frequency=100.0):
        super().__init__()
        self.base_frequency = frequency
        self.frequency_cache = {}

    def _frequency_components(self, dim, sequence_length, device, dtype):
        key = (dim, sequence_length, device, dtype)
        if key not in self.frequency_cache:
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inverse_frequency = 1.0 / self.base_frequency**exponents
            positions = torch.arange(
                sequence_length, device=device, dtype=inverse_frequency.dtype
            )
            angles = torch.outer(positions, inverse_frequency).to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            self.frequency_cache[key] = (angles.cos(), angles.sin())
        return self.frequency_cache[key]

    @staticmethod
    def _rotate(x):
        first, second = x.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def _apply_1d(self, tokens, positions, cosine, sine):
        cosine = F.embedding(positions, cosine)[:, None]
        sine = F.embedding(positions, sine)[:, None]
        return tokens * cosine + self._rotate(tokens) * sine

    def forward(self, tokens, positions):
        if tokens.shape[-1] % 4:
            raise ValueError("RoPE head dimension must be divisible by four")
        if positions.ndim != 3 or positions.shape[-1] != 2:
            raise ValueError("positions must have shape [B,N,2]")
        if positions.shape[:2] != (tokens.shape[0], tokens.shape[2]):
            raise ValueError("positions must match the token batch and sequence")

        spatial_dim = tokens.shape[-1] // 2
        maximum_position = int(positions.max()) + 1
        cosine, sine = self._frequency_components(
            spatial_dim,
            maximum_position,
            tokens.device,
            tokens.dtype,
        )
        vertical, horizontal = tokens.chunk(2, dim=-1)
        vertical = self._apply_1d(
            vertical, positions[..., 0], cosine, sine
        )
        horizontal = self._apply_1d(
            horizontal, positions[..., 1], cosine, sine
        )
        return torch.cat((vertical, horizontal), dim=-1)
