from .latent_fuse import LatentFuseNeck


def build_neck(name: str, **kwargs):
    if name != "LatentFuseNeck":
        raise ValueError(f"Unknown neck: {name}")
    return LatentFuseNeck(**kwargs)


__all__ = ["LatentFuseNeck", "build_neck"]
