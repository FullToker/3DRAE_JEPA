"""Probe memory from frozen VGGT patch tokens.

VGGT's aggregator (DINOv2-L patch embed + 24 frame / global attention rounds)
gets the 8 views without poses; the first view is its reference frame, as in the
probe's ground truth. Memory = last-round patch tokens (frame and global halves
concatenated, 2048-d), and, exactly as for the DINOv2 control, each patch's
Plucker ray from the ground-truth cameras plus a view index embedding. So every
memory in the comparison is given the same camera information.
"""

import torch

from jepa3d.paths import VJEPA_ROOT
from jepa3d.geo_probe.dino_memory import PatchMemory

VGGT_CKPT = VJEPA_ROOT / "ckpts" / "vggt.pt"
VGGT_DIM = 2048
VGGT_PATCH = 14


def load_vggt(device):
    """Frozen VGGT-1B aggregator; kept outside the trainable module so it is never saved."""
    from fusion_gv.encoders import FrozenVGGT

    return FrozenVGGT(str(VGGT_CKPT)).to(device)


class VggtPatchMemory(PatchMemory):
    def __init__(self, max_views: int = 32):
        super().__init__(VGGT_DIM, max_views)

    def forward(self, vggt, images, cams):
        """images (B, V, 3, H, W) in [0, 1], cams (B, V, 25) -> tokens (B, V * P, 2048)."""
        height, width = images.shape[-2:]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            feats = vggt(images)[-1].float()                                # (B, V, P, 2048)
        if feats.shape[2] != (height // VGGT_PATCH) * (width // VGGT_PATCH):
            raise ValueError(f"unexpected VGGT patch count {feats.shape[2]} for {height}x{width}")
        return self.embed(feats, cams, height, width, VGGT_PATCH)
