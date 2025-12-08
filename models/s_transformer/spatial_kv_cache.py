from typing import Tuple, Optional

import torch
from torch import Tensor

from utils.logger import setup_logger
logger = setup_logger(name="cache_logger", log_file="inference.log")

class SpatialKVCache:
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        max_batch_size: int,
        max_patches: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_batch_size = max_batch_size
        self.max_patches = max_patches
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype

        # Initialize values
        self.cache = None
        self.batch_size = None
        self.current_patches = None

    def initialize(self, batch_size: int) -> None:
        """Initialize cache using batch size.
        
        Args:
            batch_size: Number of examples being processed in parallel.
        """
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"got batch_size {batch_size}, with max_batch_size {self.max_batch_size}"
            )
        self.batch_size = batch_size
        self.current_patches = 0

        # Initialize cache
        self.cache = [
            {
                "k": torch.ones((
                    batch_size,
                    self.num_heads,
                    self.max_patches,
                    self.head_dim
                ), device=self.device, dtype=self.dtype),
                "v": torch.ones((
                    batch_size,
                    self.num_heads,
                    self.max_patches,
                    self.head_dim
                ), device=self.device, dtype=self.dtype)
            }
            for _ in range(self.num_layers)
        ]

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> None:
        """Update KV cache using new KV tensors.
        
        Args:
            layer_idx (int): Layer to update KV's with respect to.
            k (Tensor): Key tensor.
            v (Tensor): Value tensor.
        """
        # Initialize cache
        if self.cache is None or self.batch_size is None:
            self.initialize(k.size(0))

        # Get new spatial patches
        new_patches = k.size(2)

        # Check if cache has space
        if self.current_patches + new_patches > self.max_patches:
            current_space = self.max_patches - self.current_patches
            if self.current_patches <= 0:
                logger.info("Cache space full.")
                return
            
            # Truncate to current_space if there is space
            k = k[:, :, :current_space]
            v = v[:, :, :current_space]
            new_patches = current_space
            logger.info(f"Truncated {self.max_patches-current_space} patches.")

        # Update cache with new patches
        self.cache[layer_idx]["k"][:, :, self.current_patches:self.current_patches+new_patches] = k
        self.cache[layer_idx]["v"][:, :, self.current_patches:self.current_patches+new_patches] = v

        self.current_patches += new_patches

    def get_cached_kv(
        self, 
        layer_idx: int, 
        patches: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        """Get cached KV tensors for some number of patches.
        
        Args:
            layer_idx (int): Layer to update KV's with respect to.
            patches (Optional[int]): Number of patches to get in cache. 

        Returns:
            Tuple[Tensor, Tensor]:
                - Tensor: Key tensor.
                - Tensor: Value tensor.
        """
        if self.cache is None:
            return None, None
        
        if patches is None:
            patches = self.current_patches

        if patches > self.current_patches:
            return None, None
        
        return (
            self.cache[layer_idx]["k"][:, :, :patches],
            self.cache[layer_idx]["v"][:, :, :patches]
        )

    def reset(self) -> None:
        """Set all states to None."""
        self.cache = None
        self.batch_size = None
        self.current_patches = None
