from typing import Optional, Tuple

import torch
from torch import Tensor

class KVCache:
    """KV caching module.
    
    Args:
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        head_dim (int): Dimensionality of each attention head.
        max_batch_size (int): Maximum batch size to allocate for.
        max_seq_len (int): Maximum sequence length to allocate for.
        device (torch.device): Accelerator at use.
        dtype (torch.dtype): Data type of model parameters.
    """
    def __init__(
        self,
        num_heads: int,
        num_layers: int,
        head_dim: int,
        max_batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ):
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        # Initialize states to None
        self.batch_size = None
        self.cache = None
        self.current_seq_len = None

    def initialize(self, batch_size: int) -> None:
        """Initialize cache for given batch size.
        
        Args:
            batch_size (int): Batch size at a given timestep.
        """
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"got batch_size {batch_size} > max_batch_size {self.max_batch_size}"
            )
        self.batch_size = batch_size
        self.current_seq_len = 0

        self.cache = [
            {
                'k': torch.ones((
                    batch_size, self.num_heads, self.max_seq_len, self.head_dim
                ), device=self.device, dtype=self.dtype),
                'v': torch.ones((
                    batch_size, self.num_heads, self.max_seq_len, self.head_dim
                ), device=self.device, dtype=self.dtype)
            }
            for _ in range(self.num_layers)
        ]

    def update(self, layer_idx: int, k: Tensor, v: int) -> None:
        """Update KV cache using past KVs.
        
        Args:
            layer_idx (int): Current layer to update KVs with respect to.
            k (Tensor): Key tensor.
            v (Tensor): Value tensor.
        """
        if self.cache is None or self.batch_size is None:
            self.initialize(k.size(0))

        # Get new tokens
        new_seq_len = k.size(2)

        # Truncate if needed
        if self.current_seq_len + new_seq_len > self.max_seq_len:
            current_space = self.max_seq_len - self.current_seq_len
            if current_space <= 0:
                return
            
            # Truncate to current_space
            k = k[:, :, :current_space]
            v = v[:, :, :current_space]
            new_seq_len = current_space

        # Update sequence length dimension with new tokens
        self.cache[layer_idx]["k"][:, :, self.current_seq_len:self.current_seq_len+new_seq_len] = k
        self.cache[layer_idx]["v"][:, :, self.current_seq_len:self.current_seq_len+new_seq_len] = v

        self.current_seq_len += new_seq_len

    def get(
        self, 
        layer_idx: int, 
        tokens: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        """Get cached KV tensors up to requested token amount.
        
        Args:
            layer_idx (int): Current layer to update KVs with respec to.
            tokens (Optional[int]): Number of tokens to receive KV tensors with.

        Returns:
            Tuple[Tensor, Tensor]:
                - Tensor: Key tensor with requested token amount.
                - Tensor: Value tensor with requested token amount.
        """
        if self.cache is None:
            return None, None
        
        if tokens is None:
            tokens = self.current_seq_len

        if tokens > self.current_seq_len:
            return None, None
        
        return (
            self.cache[layer_idx]["k"][:, :, :tokens],
            self.cache[layer_idx]["v"][:, :, :tokens]
        )

    def reset(self) -> None:
        """Reset cache states to None."""
        self.cache = None
        self.batch_size = None
        self.current_seq_len = None
    