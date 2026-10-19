import torch
import torch.nn.functional as F


class DiffAug:
    def __init__(self, prob=1.0, cutout=0.2):
        if not 0 <= prob <= 1:
            raise ValueError("Augmentation probability must be in [0, 1]")
        if not 0 <= cutout <= 1:
            raise ValueError("Cutout ratio must be in [0, 1]")
        self.grids = {}
        self.prob = prob
        self.using_cutout = prob > 0 and cutout > 0
        self.cutout = cutout

    def __str__(self):
        cutout = self.cutout if self.using_cutout else 0
        return f"DiffAug(p={self.prob}, cutout={cutout})"

    def get_grids(self, batch_size, height, width, device):
        key = (batch_size, height, width, device)
        if key not in self.grids:
            self.grids[key] = torch.meshgrid(
                torch.arange(batch_size, dtype=torch.long, device=device),
                torch.arange(height, dtype=torch.long, device=device),
                torch.arange(width, dtype=torch.long, device=device),
                indexing="ij",
            )
        return self.grids[key]

    def aug(self, images: torch.Tensor) -> torch.Tensor:
        if images.dtype != torch.float32:
            images = images.float()
        if self.prob == 0:
            return images

        translate, color, cutout = torch.rand(3) <= self.prob
        translate = translate.item()
        color = color.item()
        cutout = cutout.item()
        batch_size, device = images.shape[0], images.device
        random_values = torch.rand(7, batch_size, 1, 1, device=device)
        height, width = images.shape[-2:]

        if translate:
            delta_height = round(height * 0.125)
            delta_width = round(width * 0.125)
            translation_height = (
                random_values[0]
                .mul(2 * delta_height + 1)
                .floor()
                .long()
                - delta_height
            )
            translation_width = (
                random_values[1]
                .mul(2 * delta_width + 1)
                .floor()
                .long()
                - delta_width
            )
            grid_batch, grid_height, grid_width = self.get_grids(
                batch_size, height, width, device
            )
            grid_height = (grid_height + translation_height).add_(1).clamp_(
                0, height + 1
            )
            grid_width = (grid_width + translation_width).add_(1).clamp_(
                0, width + 1
            )
            padded = F.pad(images, [1, 1, 1, 1, 0, 0, 0, 0])
            images = padded.permute(0, 2, 3, 1).contiguous()[
                grid_batch, grid_height, grid_width
            ]
            images = images.permute(0, 3, 1, 2).contiguous()

        if color:
            images = images.add(random_values[2].unsqueeze(-1).sub(0.5))
            channel_mean = images.mean(dim=1, keepdim=True)
            images = images.sub(channel_mean).mul(
                random_values[3].unsqueeze(-1).mul(2)
            ).add_(channel_mean)
            image_mean = images.mean(dim=(1, 2, 3), keepdim=True)
            images = images.sub(image_mean).mul(
                random_values[4].unsqueeze(-1).add(0.5)
            ).add_(image_mean)

        if self.using_cutout and cutout:
            cutout_height = round(height * self.cutout)
            cutout_width = round(width * self.cutout)
            offset_height = (
                random_values[5]
                .mul(height + (1 - cutout_height % 2))
                .floor()
                .long()
            )
            offset_width = (
                random_values[6]
                .mul(width + (1 - cutout_width % 2))
                .floor()
                .long()
            )
            grid_batch, grid_height, grid_width = self.get_grids(
                batch_size, cutout_height, cutout_width, device
            )
            grid_height = (grid_height + offset_height).sub_(
                cutout_height // 2
            ).clamp(min=0, max=height - 1)
            grid_width = (grid_width + offset_width).sub_(
                cutout_width // 2
            ).clamp(min=0, max=width - 1)
            mask = torch.ones(
                batch_size, height, width, dtype=images.dtype, device=device
            )
            mask[grid_batch, grid_height, grid_width] = 0
            images = images.mul(mask.unsqueeze(1))

        return images
