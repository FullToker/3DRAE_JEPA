from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from utils.weights import load_safetensors, load_tensor_dict

from .decoders import build_decoder
from .encoders import build_encoder
from .neck import build_neck

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


class RAE_Latent(nn.Module):
    def __init__(
        self,
        encoder_cls: str,
        encoder_params: Optional[dict] = None,
        neck_cls: str = "LatentFuseNeck",
        neck_params: Optional[dict] = None,
        decoder_img_cls: Optional[str] = None,
        decoder_img_params: Optional[dict] = None,
        decoder_pmap_cls: Optional[str] = None,
        decoder_pmap_params: Optional[dict] = None,
        pretrained_neck_path: Optional[str] = None,
        pretrained_decoder_img_path: Optional[str] = None,
        pretrained_decoder_pmap_path: Optional[str] = None,
        noise_tau: float = 0.0,
        normalization_stat_path: Optional[str] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        if encoder_cls not in {"Dinov2withNorm", "SigLIP2wNorm"}:
            raise ValueError(f"Unsupported release encoder: {encoder_cls}")

        self.encoder_type = encoder_cls
        encoder_params = dict(encoder_params or {})
        self.encoder = build_encoder(encoder_cls, **encoder_params)
        self.encoder.requires_grad_(False)
        self.encoder_patch_size = self.encoder.patch_size

        neck_params = dict(neck_params or {})
        neck_params["hidden_size_input"] = self.encoder.hidden_size
        self.neck = build_neck(neck_cls, **neck_params)

        if decoder_img_cls is not None:
            self.decoder_img = build_decoder(
                decoder_img_cls, **dict(decoder_img_params or {})
            )
        else:
            self.decoder_img = None
        if decoder_pmap_cls is not None:
            self.decoder_pmap = build_decoder(
                decoder_pmap_cls, **dict(decoder_pmap_params or {})
            )
        else:
            self.decoder_pmap = None
        if pretrained_neck_path is not None:
            load_safetensors(self.neck, pretrained_neck_path, strict=True)
        if pretrained_decoder_img_path is not None:
            if self.decoder_img is None:
                raise ValueError("decoder_img weights were given without decoder_img_cls")
            load_safetensors(
                self.decoder_img, pretrained_decoder_img_path, strict=True
            )
        if pretrained_decoder_pmap_path is not None:
            if self.decoder_pmap is None:
                raise ValueError("point decoder weights were given without decoder_pmap_cls")
            load_safetensors(
                self.decoder_pmap, pretrained_decoder_pmap_path, strict=True
            )

        if encoder_cls == "SigLIP2wNorm":
            mean, std = SIGLIP_MEAN, SIGLIP_STD
        else:
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        self.register_buffer(
            "image_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False
        )

        self.noise_tau = noise_tau
        self.eps = eps
        if normalization_stat_path is not None:
            stats = load_tensor_dict(normalization_stat_path)
            if set(stats) != {"mean", "var"}:
                raise KeyError(
                    "Normalization stats must contain exactly the 'mean' and 'var' tensors"
                )
            self.register_buffer("latent_mean", stats["mean"], persistent=False)
            self.register_buffer("latent_var", stats["var"], persistent=False)
            self.do_normalization = True
        else:
            self.register_buffer("latent_mean", None, persistent=False)
            self.register_buffer("latent_var", None, persistent=False)
            self.do_normalization = False

    def noising(self, x: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand(
            (x.size(0),) + (1,) * (x.ndim - 1), device=x.device
        )
        return x + noise_sigma * torch.randn_like(x)

    @staticmethod
    def _as_multiview(x: torch.Tensor):
        if x.ndim == 4:
            return x.unsqueeze(1)
        if x.ndim != 5:
            raise ValueError(f"Expected BCHW or BSCHW images, got {tuple(x.shape)}")
        return x

    @torch.no_grad()
    def _encode_visible(self, images, visible_ids):
        batch_size, views, _, height, width = images.shape
        if visible_ids is None:
            visible_images = images.reshape(batch_size * views, 3, height, width)
            visible_views = views
        else:
            visible_views = visible_ids.shape[1]
            if visible_views == 0:
                return None
            batch_ids = torch.arange(batch_size, device=images.device).view(-1, 1)
            visible_images = images[batch_ids, visible_ids].reshape(
                batch_size * visible_views, 3, height, width
            )
        features = self.encoder(visible_images)
        return features.reshape(batch_size, visible_views, *features.shape[1:])

    def _encode_dinov2(self, images, visible_ids, ray_maps):
        normalized = (images - self.image_mean.unsqueeze(1)) / self.image_std.unsqueeze(1)
        features = self._encode_visible(normalized, visible_ids)
        return self.neck(features, visible_ids, ray_maps)

    def _encode_siglip2(self, images, visible_ids, ray_maps):
        batch_size, views, _, height, width = images.shape
        input_patch_size = self.neck.patch_size
        resized_height = height // input_patch_size * self.encoder_patch_size
        resized_width = width // input_patch_size * self.encoder_patch_size
        resized = F.interpolate(
            images.reshape(batch_size * views, 3, height, width),
            size=(resized_height, resized_width),
            mode="bicubic",
            align_corners=False,
        ).reshape(batch_size, views, 3, resized_height, resized_width)
        normalized = (
            resized - self.image_mean.unsqueeze(1)
        ) / self.image_std.unsqueeze(1)
        features = self._encode_visible(normalized, visible_ids)
        return self.neck(features, visible_ids, ray_maps)

    def encode(
        self,
        x: torch.Tensor,
        visible_ids: Optional[torch.Tensor] = None,
        ray_maps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        images = self._as_multiview(x)
        if ray_maps is None:
            raise ValueError("ray_maps is required")
        if self.encoder_type == "SigLIP2wNorm":
            z = self._encode_siglip2(images, visible_ids, ray_maps)
        else:
            z = self._encode_dinov2(images, visible_ids, ray_maps)
        if self.training and self.noise_tau > 0:
            z = self.noising(z)
        return self._apply_norm(z) if self.do_normalization else z

    def _apply_norm(self, z):
        return (z - self.latent_mean) / torch.sqrt(self.latent_var + self.eps)

    def _unapply_norm(self, z):
        return z * torch.sqrt(self.latent_var + self.eps) + self.latent_mean

    def decode(self, z: torch.Tensor, ray_maps: torch.Tensor):
        if self.do_normalization:
            z = self._unapply_norm(z)
        if self.decoder_img is not None:
            image_output = self.decoder_img(z, ray_maps)
            images = self.decoder_img.unpatchify(
                image_output.logits, original_image_size=ray_maps.shape[-3:-1]
            )
        else:
            images = None
        if self.decoder_pmap is not None:
            point_output = self.decoder_pmap(z, ray_maps)
            points = point_output.pts3d
            confidence = point_output.pts3d_conf
        else:
            points = confidence = None
        return images, points, confidence

    def forward(
        self,
        x: torch.Tensor,
        visible_ids: Optional[torch.Tensor] = None,
        ray_maps: Optional[torch.Tensor] = None,
        ray_maps_nv: Optional[torch.Tensor] = None,
    ):
        z = self.encode(x, visible_ids=visible_ids, ray_maps=ray_maps)
        if ray_maps_nv is None:
            return z
        return self.decode(z, ray_maps_nv)
