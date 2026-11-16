"""Point prompts anywhere on the object, not only at its projected centre.

The point probe was trained with the centre of each object's box as the prompt.
A patch picked by name (dino.txt) lands somewhere on the object, where that
probe is worse (0.47 m against 0.27 m). This dataset moves a share of the
prompts to a random patch whose ray passes through the object's 3D box.
"""

import random

import torch

from jepa3d.geo_probe.data import IMAGE_SIZE
from jepa3d.tools.check_teacher_grounding import PATCH
from jepa3d.tools.diagnose_teacher_grounding import BoxDataset, patch_rays, rays_hit_boxes
from jepa3d.x_encoder import unpack_cams


class BoxPromptDataset(BoxDataset):
    """share: probability that an object's prompt is a random in-box patch (a
    random pixel of it) instead of the projected centre. Objects whose centre is
    outside every view get an in-box prompt whenever a ray passes through the box."""

    def __init__(self, *args, share: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.share = share

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        K, w2c = unpack_cams(sample["cams"])
        origins, directions = patch_rays(K.float(), w2c.float(), IMAGE_SIZE)
        in_box = rays_hit_boxes(origins * sample["scale"], directions, sample.pop("box_center"),
                                sample.pop("box_half"), sample.pop("box_rotation")).flatten(1)       # (O, V*h*w)
        h, w = IMAGE_SIZE[0] // PATCH, IMAGE_SIZE[1] // PATCH
        for o in range(len(in_box)):
            inside = in_box[o].nonzero().flatten()
            if len(inside) and (sample["prompt"][o, 0] < 0 or random.random() < self.share):
                view, rest = divmod(int(inside[random.randrange(len(inside))]), h * w)
                sample["prompt"][o] = torch.tensor([view, (rest % w + random.random()) * PATCH,
                                                    (rest // w + random.random()) * PATCH])
        return sample
