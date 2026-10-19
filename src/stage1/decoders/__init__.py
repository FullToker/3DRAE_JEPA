from .latent_image_decoder import LatentImageDecoder
from .latent_point_decoder import LatentPointDecoder


def build_decoder(name: str, **kwargs):
    decoders = {
        "LatentImageDecoder": LatentImageDecoder,
        "LatentPointDecoder": LatentPointDecoder,
    }
    try:
        return decoders[name](**kwargs)
    except KeyError as error:
        raise ValueError(f"Unknown decoder: {name}") from error


__all__ = [
    "LatentImageDecoder",
    "LatentPointDecoder",
    "build_decoder",
]
