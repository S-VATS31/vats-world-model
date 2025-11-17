from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast

class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x: Tensor):
        with autocast(device_type=x.device.type):
            pass
