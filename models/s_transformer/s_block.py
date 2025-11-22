from typing import *

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast


class SpatialTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x: Tensor) -> Tensor:
        with autocast(device_type=x.device.type):
            pass
