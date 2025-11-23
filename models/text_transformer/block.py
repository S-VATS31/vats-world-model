from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast

from models.mlp import MLPBlock
from models.text_transformer.attention import CausalAttentionBlock

class TextTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x: Tensor) -> Tensor:
        with autocast(device_type=x.device.type):
            pass
