from typing import Tuple, Optional

import torch
from torch import Tensor

from utils.logger import setup_logger
logger = setup_logger(name="cache_logger", log_file="inference.log")

class TemporalKVCache:
    """KV cache for temporal attention.
    
    Args:
        max_batch_size (int): Number of examples being processed in parallel.
        max_frames (int): Maximum number of input frames allowed.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimensionality of each attention head.
        num_layers (int): Number of transformer blocks to be stacked.
        device (torch.device): Accelerator at use.
        dtype (torch.dtype): Data type of tensors.
    """
    def __init__(
        self,
        max_batch_size: int,
        max_frames: int,
        num_heads: int,
        head_dim: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype
    ):
        self.max_batch_size = max_batch_size
        self.max_frames = max_frames
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype

        # Initialize to None
        self.cache = None
        self.batch_size = None
        self.current_frames = None

    def initialize(self, batch_size: int, num_spatial_patches: int) -> None:
        """Initialize cache.
        
        Args:
            batch_size (int): Number of examples being processed in parallel.
            num_spatial_patches (int): Number of spatial patches to initialize for.
        """
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"batch_size {batch_size} exceeded max_batch_size {self.max_batch_size}"
            )
        self.batch_size = batch_size
        self.current_frames = 0

        self.cache = [
            {
                "k": torch.zeros((
                        batch_size*num_spatial_patches, 
                        self.num_heads,
                        self.max_frames, 
                        self.head_dim
                    ), device=self.device, dtype=self.dtype),
                "v": torch.zeros((
                        batch_size*num_spatial_patches, 
                        self.num_heads,
                        self.max_frames, 
                        self.head_dim
                    ), device=self.device, dtype=self.dtype)
            }
            for _ in range(self.num_layers)
        ]

    def update(
        self, 
        layer_idx: int, 
        k: Tensor, 
        v: Tensor,
        num_spatial_patches: int
    ) -> None:
        """Update KV cache with new KV tensors only.
        
        Args:
            layer_idx (int): Current layer to update KVs with respect to.
            k (Tensor): New key tensor to add to cache.
            v (Tensor): New value tensor to add to cache.
            num_spatial_patches (int): Number of spatial patches to get batch size.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"expected 0 < layer_idx < num_layers, got {layer_idx}")
        batch_spatial = k.size(0)
        expected_batch_size = batch_spatial // num_spatial_patches
        
        if self.cache is None or self.batch_size != expected_batch_size:
            self.initialize(expected_batch_size, num_spatial_patches)

        # Get new sequence length
        new_frames = k.size(2)
        
        # Check if we have space
        if self.current_frames + new_frames > self.max_frames:
            current_space = self.max_frames - self.current_frames
            if current_space <= 0:
                logger.info("Cache space full")
                return
            
            # Truncate to create space
            k = k[:, :, :current_space]
            v = v[:, :, :current_space]
            new_frames = current_space
            logger.info(f"Truncated {self.max_frames - current_space} frames")
        
        # Update cache with new tensors
        self.cache[layer_idx]["k"][:, :, self.current_frames:self.current_frames+new_frames] = k
        self.cache[layer_idx]["v"][:, :, self.current_frames:self.current_frames+new_frames] = v

        # Increment frames
        self.current_frames += new_frames

    def get_cached_kv(
        self, 
        layer_idx: int, 
        frames: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        """Get cached KV tensors.
        
        Args:
            layer_idx (int): Current layer to retrieve KVs.
            frames (int, optional): Number of frames to retrieve up to. 

        Returns:
            Tuple:
                Tensor: Key tensor.
                Tensor: Value tensor.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"expected 0 < layer_idx < num_layers, got {layer_idx}")
        
        if self.cache is None:
            return None, None
        
        # frames is None, return all current frames
        if frames is None:
            frames = self.current_frames
            
        if frames > self.current_frames:
            return None, None
        
        return (
            self.cache[layer_idx]["k"][:, :, :frames],
            self.cache[layer_idx]["v"][:, :, :frames]
        )

    def reset(self) -> None:
        """Reset cache to initial state."""
        self.cache = None
        self.current_frames = None
        self.batch_size = None
        