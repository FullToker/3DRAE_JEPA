import torch
from torch import nn
from transformers import SiglipModel

class SigLIP2wNorm(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = SiglipModel.from_pretrained(model_name).vision_model
        self.model.post_layernorm.elementwise_affine = False
        self.model.post_layernorm.weight = None
        self.model.post_layernorm.bias = None
        self.model.requires_grad_(False)
        self.hidden_size = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size

    @torch.no_grad()
    def forward(self, images):
        output = self.model(images, interpolate_pos_encoding=True)
        return output.last_hidden_state
