from typing import Optional, Tuple

import torch
from torch import Tensor

class KVCache:
    def __init__(self):
        pass

    def initialize(self, batch_size: int) -> None:
        pass

    def update(self, layer_idx: int, k: Tensor, v: int) -> None:
        pass

    def get(self, layer_idx: int, tokens: Optional[int]) -> Tuple[Tensor, Tensor]:
        pass

    def reset(self) -> None:
        pass
    