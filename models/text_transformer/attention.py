from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast

from models.rope1d import RoPE
from utils.attention_utils import extend_kv_heads, apply_qk_norm

class CausalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def _setup_qkv(self) -> Tuple[Tensor, Tensor, Tensor]:
        pass

    def forward(self, x: Tensor) -> Tensor:
        with autocast(device_type=x.device.type):
            pass


class CausalAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x: Tensor) -> Tensor:
        with autocast(device_type=x.device.type):
            pass

