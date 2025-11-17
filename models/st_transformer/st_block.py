from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast

from utils.mlp import MLPBlock
from models.st_transformer.st_attention import SpatioTemporalAttentionBlock

class SpatioTemporalTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x: Tensor) -> Tensor:
        with autocast(device_type=x.device.type):
            pass
