import torch

from .dinodisc import DinoDisc


class DinoDiscriminator(DinoDisc):
    """DINO discriminator interface for paired fake and real images."""

    def __init__(self, device: torch.device, **kwargs):
        super().__init__(device=device, **kwargs)

    def classify(self, img: torch.Tensor) -> torch.Tensor:
        return super().forward(img)

    def forward(self, fake: torch.Tensor, real: torch.Tensor | None = None):
        logits_fake = self.classify(fake)
        logits_real = self.classify(real) if real is not None else None
        return logits_fake, logits_real
