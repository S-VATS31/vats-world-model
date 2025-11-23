from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast

from models.text_transformer.kv_cache import KVCache
from models.text_transformer.block import TextTransformerBlock
from configs.text_transformer.model_args.xsmall import ModelArgs

class TestTransformer(nn.Module):
    def __init__(model_args: ModelArgs):
        super().__init__()
        pass

    def _init_weights(module) -> None:
        pass

    def forward(self, x: Tensor) -> Tensor:
        with autocast(device_type=x.device.type):
            pass
