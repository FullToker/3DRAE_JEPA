# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the repository root.

import torch
import torch.nn.functional as F
from torch import nn


def make_sincos_pos_embed(embed_dim, pos, omega_0=100):
    if embed_dim % 2:
        raise ValueError("Embedding dimension must be even")
    dtype = torch.float32 if pos.device.type == "mps" else torch.double
    omega = torch.arange(embed_dim // 2, dtype=dtype, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega
    out = torch.einsum("m,d->md", pos.reshape(-1), omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1).float()


def position_grid_to_embed(pos_grid, embed_dim, omega_0=100):
    height, width, grid_dim = pos_grid.shape
    if grid_dim != 2:
        raise ValueError("Position grid must have two coordinates")
    pos_flat = pos_grid.reshape(-1, grid_dim)
    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0)
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0)
    return torch.cat([emb_x, emb_y], dim=-1).view(height, width, embed_dim)


def create_uv_grid(width, height, aspect_ratio, dtype=None, device=None):
    diagonal = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diagonal
    span_y = 1.0 / diagonal
    x = torch.linspace(
        -span_x * (width - 1) / width,
        span_x * (width - 1) / width,
        steps=width,
        dtype=dtype,
        device=device,
    )
    y = torch.linspace(
        -span_y * (height - 1) / height,
        span_y * (height - 1) / height,
        steps=height,
        dtype=dtype,
        device=device,
    )
    uu, vv = torch.meshgrid(x, y, indexing="xy")
    return torch.stack((uu, vv), dim=-1)


def activate_head(output):
    feature_map = output.permute(0, 2, 3, 1).contiguous()
    xyz, confidence = feature_map[..., :-1], feature_map[..., -1]
    points = torch.sign(xyz) * torch.expm1(torch.abs(xyz))
    return points, 1 + confidence.exp()


def custom_interpolate(
    x, size=None, scale_factor=None, mode="bilinear", align_corners=True
):
    if size is None:
        size = (
            int(x.shape[-2] * scale_factor),
            int(x.shape[-1] * scale_factor),
        )
    max_elements = 1610612736
    output_elements = size[0] * size[1] * x.shape[0] * x.shape[1]
    if output_elements > max_elements:
        chunks = torch.chunk(x, chunks=(output_elements // max_elements) + 1, dim=0)
        x = torch.cat(
            [
                F.interpolate(
                    chunk, size=size, mode=mode, align_corners=align_corners
                )
                for chunk in chunks
            ],
            dim=0,
        )
        return x.contiguous()
    return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)


class ResidualConvUnit(nn.Module):
    def __init__(self, features, activation):
        super().__init__()
        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1
        )
        self.activation = activation

    def forward(self, x):
        output = self.conv1(self.activation(x))
        output = self.conv2(self.activation(output))
        return output + x


class FeatureFusionBlock(nn.Module):
    def __init__(
        self,
        features,
        activation,
        align_corners=True,
        has_residual=True,
    ):
        super().__init__()
        self.align_corners = align_corners
        self.out_conv = nn.Conv2d(
            features, features, kernel_size=1, stride=1
        )
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation)
        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation)

    def forward(self, output, residual=None, size=None):
        if self.has_residual:
            output = output + self.resConfUnit1(residual)
        output = self.resConfUnit2(output)
        resize = {"scale_factor": 2} if size is None else {"size": size}
        output = custom_interpolate(
            output, **resize, mode="bilinear", align_corners=self.align_corners
        )
        return self.out_conv(output)


def _make_fusion_block(features, has_residual=True):
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        align_corners=True,
        has_residual=has_residual,
    )


def _make_scratch(in_shape, out_shape):
    scratch = nn.Module()
    channels = [out_shape, out_shape, out_shape, out_shape]
    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], channels[0], 3, stride=1, padding=1, bias=False
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], channels[1], 3, stride=1, padding=1, bias=False
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], channels[2], 3, stride=1, padding=1, bias=False
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3],
            channels[3],
            3,
            stride=1,
            padding=1,
            bias=False,
        )
    return scratch


class DPTHead(nn.Module):
    def __init__(
        self,
        dim_in,
        patch_size=14,
    ):
        super().__init__()
        features = 256
        out_channels = (256, 512, 1024, 1024)
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList(
            [nn.Conv2d(dim_in, channels, 1) for channels in out_channels]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(out_channels[0], out_channels[0], 4, stride=4),
                nn.ConvTranspose2d(out_channels[1], out_channels[1], 2, stride=2),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], 3, stride=2, padding=1),
            ]
        )
        self.scratch = _make_scratch(out_channels, features)
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(
            features, has_residual=False
        )
        self.scratch.output_conv1 = nn.Conv2d(
            features, features // 2, 3, padding=1
        )
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, 1),
        )

    def forward(
        self,
        aggregated_tokens_list,
        input_shape,
        frames_chunk_size=8,
    ):
        _, views, _, _ = input_shape
        if frames_chunk_size >= views:
            return self._forward_impl(aggregated_tokens_list, input_shape)
        outputs, confidences = [], []
        for start in range(0, views, frames_chunk_size):
            end = min(start + frames_chunk_size, views)
            result = self._forward_impl(
                aggregated_tokens_list,
                input_shape,
                start,
                end,
            )
            prediction, confidence = result
            outputs.append(prediction)
            confidences.append(confidence)
        return torch.cat(outputs, dim=1), torch.cat(confidences, dim=1)

    def _forward_impl(
        self,
        aggregated_tokens_list,
        input_shape,
        frames_start_idx=None,
        frames_end_idx=None,
    ):
        batch_size, views, height, width = input_shape
        if frames_start_idx is not None:
            views = frames_end_idx - frames_start_idx
        patch_size = self.patch_size
        patch_height, patch_width = height // patch_size, width // patch_size
        features = []
        if len(aggregated_tokens_list) != 4:
            raise ValueError("DPT head expects four intermediate feature maps")
        for index, x in enumerate(aggregated_tokens_list):
            if frames_start_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]
            x = self.norm(x.reshape(batch_size * views, -1, x.shape[-1]))
            x = x.permute(0, 2, 1).reshape(
                x.shape[0], x.shape[-1], patch_height, patch_width
            ).contiguous()
            x = self.projects[index](x)
            x = self._apply_pos_embed(x, width, height)
            features.append(self.resize_layers[index](x))

        output = self.scratch_forward(features)
        output = custom_interpolate(
            output,
            (
                patch_height * patch_size,
                patch_width * patch_size,
            ),
            mode="bilinear",
            align_corners=True,
        )
        output = self._apply_pos_embed(output, width, height)
        output = self.scratch.output_conv2(output)
        points, confidence = activate_head(output)
        return (
            points.view(batch_size, views, *points.shape[1:]),
            confidence.view(batch_size, views, *confidence.shape[1:]),
        )

    def _apply_pos_embed(self, x, width, height, ratio=0.1):
        patch_width, patch_height = x.shape[-1], x.shape[-2]
        position = create_uv_grid(
            patch_width,
            patch_height,
            aspect_ratio=width / height,
            dtype=x.dtype,
            device=x.device,
        )
        position = position_grid_to_embed(position, x.shape[1]) * ratio
        position = position.permute(2, 0, 1)[None].expand(
            x.shape[0], -1, -1, -1
        ).contiguous()
        return x + position

    def scratch_forward(self, features):
        layer_1, layer_2, layer_3, layer_4 = features
        layer_1 = self.scratch.layer1_rn(layer_1)
        layer_2 = self.scratch.layer2_rn(layer_2)
        layer_3 = self.scratch.layer3_rn(layer_3)
        layer_4 = self.scratch.layer4_rn(layer_4)
        output = self.scratch.refinenet4(layer_4, size=layer_3.shape[2:])
        output = self.scratch.refinenet3(output, layer_3, size=layer_2.shape[2:])
        output = self.scratch.refinenet2(output, layer_2, size=layer_1.shape[2:])
        output = self.scratch.refinenet1(output, layer_1)
        return self.scratch.output_conv1(output)
