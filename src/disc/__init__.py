import torch

from .diffaug import DiffAug
from .discriminator import DinoDiscriminator
from .gan_loss import hinge_d_loss, vanilla_d_loss, vanilla_g_loss
from .lpips import LPIPS


def build_discriminator(config: dict, device: torch.device):
    architecture = config["architecture"]
    discriminator = DinoDiscriminator(
        device=device,
        dino_ckpt_path=architecture["weights"],
        ks=int(architecture.get("kernel_size", 9)),
    ).to(device)
    augment = config.get("augment", {})
    return discriminator, DiffAug(
        prob=float(augment.get("probability", 1.0)),
        cutout=float(augment.get("cutout", 0.0)),
    )


__all__ = [
    "LPIPS",
    "DiffAug",
    "DinoDiscriminator",
    "hinge_d_loss",
    "vanilla_d_loss",
    "vanilla_g_loss",
    "build_discriminator",
]
