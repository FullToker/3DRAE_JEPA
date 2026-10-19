import torch
from torch import nn
from transformers import Dinov2WithRegistersModel

class Dinov2withNorm(nn.Module):
    def __init__(
        self,
        dinov2_path: str,
        normalize: bool = True,
    ):
        super().__init__()
        try:
            self.encoder = Dinov2WithRegistersModel.from_pretrained(
                dinov2_path, local_files_only=True
            )
        except (OSError, ValueError, AttributeError):
            self.encoder = Dinov2WithRegistersModel.from_pretrained(
                dinov2_path, local_files_only=False
            )
        self.encoder.requires_grad_(False)
        if normalize:
            self.encoder.layernorm.elementwise_affine = False
            self.encoder.layernorm.weight = None
            self.encoder.layernorm.bias = None
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.encoder(x)
        return output.last_hidden_state[:, 5:]
