from collections import namedtuple
from pathlib import Path

import torch
from torch import nn
from torchvision import models


class ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "shift", torch.tensor([-.030, -.088, -.188])[None, :, None, None]
        )
        self.register_buffer(
            "scale", torch.tensor([.458, .448, .450])[None, :, None, None]
        )

    def forward(self, tensor):
        return (tensor - self.shift) / self.scale


class NetLinLayer(nn.Module):
    def __init__(self, channels, use_dropout=True):
        super().__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers.append(nn.Conv2d(channels, 1, 1, bias=False))
        self.model = nn.Sequential(*layers)

    def forward(self, tensor):
        return self.model(tensor)


class VGG16FeatureExtractor(nn.Module):
    def __init__(self, weights_path: str | Path):
        super().__init__()
        model = models.vgg16(weights=None)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        features = model.features
        self.slice1 = nn.Sequential(*features[:4])
        self.slice2 = nn.Sequential(*features[4:9])
        self.slice3 = nn.Sequential(*features[9:16])
        self.slice4 = nn.Sequential(*features[16:23])
        self.slice5 = nn.Sequential(*features[23:30])
        self.requires_grad_(False)

    def forward(self, tensor):
        relu1_2 = self.slice1(tensor)
        relu2_2 = self.slice2(relu1_2)
        relu3_3 = self.slice3(relu2_2)
        relu4_3 = self.slice4(relu3_3)
        relu5_3 = self.slice5(relu4_3)
        outputs = namedtuple(
            "VggOutputs",
            ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"],
        )
        return outputs(relu1_2, relu2_2, relu3_3, relu4_3, relu5_3)


def _normalize(tensor, eps=1e-10):
    return tensor / (torch.sum(tensor**2, dim=1, keepdim=True).sqrt() + eps)


class LPIPS(nn.Module):
    def __init__(
        self,
        vgg_weights_path: str | Path,
        lpips_weights_path: str | Path,
        use_dropout: bool = True,
    ):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.net = VGG16FeatureExtractor(vgg_weights_path)
        channels = [64, 128, 256, 512, 512]
        self.linear = nn.ModuleList(
            [NetLinLayer(value, use_dropout=use_dropout) for value in channels]
        )
        state = torch.load(lpips_weights_path, map_location="cpu", weights_only=True)
        legacy_names = {f"lin{i}.model.1.weight": f"linear.{i}.model.1.weight" for i in range(5)}
        legacy_names.update(
            {f"lin{i}.model.0.weight": f"linear.{i}.model.0.weight" for i in range(5)}
        )
        state = {legacy_names.get(name, name): value for name, value in state.items()}
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing = [
            name
            for name in missing
            if not name.startswith("net.")
            and not name.startswith("scaling_layer.")
        ]
        if missing or unexpected:
            raise RuntimeError(
                f"Invalid LPIPS weights: missing={missing}, unexpected={unexpected}"
            )
        self.requires_grad_(False)
        self.eval()

    def forward(self, input, target):
        input_features = self.net(self.scaling_layer(input))
        target_features = self.net(self.scaling_layer(target))
        value = input.new_zeros(input.shape[0], 1, 1, 1)
        for layer, input_feature, target_feature in zip(
            self.linear, input_features, target_features
        ):
            difference = (_normalize(input_feature) - _normalize(target_feature)) ** 2
            value = value + layer(difference).mean((2, 3), keepdim=True)
        return value.mean()
