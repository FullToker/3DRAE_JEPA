from .dinov2 import Dinov2withNorm
from .siglip2 import SigLIP2wNorm


def build_encoder(name: str, **kwargs):
    encoders = {
        "Dinov2withNorm": Dinov2withNorm,
        "SigLIP2wNorm": SigLIP2wNorm,
    }
    try:
        return encoders[name](**kwargs)
    except KeyError as error:
        raise ValueError(f"Unknown encoder: {name}") from error


__all__ = ["Dinov2withNorm", "SigLIP2wNorm", "build_encoder"]
