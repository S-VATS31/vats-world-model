from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast

from utils.mlp import MLPBlock
from models.s_transformer.spatial_kv_cache import SpatialKVCache
from models.s_transformer.s_attention import SpatialAttentionBlock

class SpatialTransformerBlock(nn.Module):
    """Spatial transformer block.
    
    Args:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads.
        query_groups (int): Query groups for GQA.
        rope_theta (float): Exponential base for RoPE.
        softmax_scale (float): Softmax scale for attention computation.
        use_qkv_bias (bool): Whether to use QKV bias.
        use_o_bias (bool): Whether to use output bias.
        use_qkv_proj (bool): Whether to use QKV projection or not.
        rms_norm_eps (float): Epsilon value for RMSNorm.
        dropout_prob (float): Dropout probability.
        d_ffn (int): Dimensionality MLP.
        use_mlp_bias (bool): Whether to use MLP bias or not.
        device (torch.device): Accelerator at use.
        dtype (torch.dtype): Data type of model parameters.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        rope_theta: float,
        softmax_scale: float,
        use_qkv_bias: bool,
        use_o_bias: bool,
        use_qkv_proj: bool,
        rms_norm_eps: float,
        dropout_prob: float,
        d_ffn: int,
        use_mlp_bias: bool,
        device: torch.device,
        dtype: torch.dtype
    ):
        super().__init__()
        
        self.attn_block = SpatialAttentionBlock(
            d_model=d_model,
            num_heads=num_heads,
            query_groups=query_groups,
            rope_theta=rope_theta,
            softmax_scale=softmax_scale,
            use_qkv_bias=use_qkv_bias,
            use_o_bias=use_o_bias,
            use_qkv_proj=use_qkv_proj,
            rms_norm_eps=rms_norm_eps,
            dropout_prob=dropout_prob,
            device=device,
            dtype=dtype
        )
        self.mlp_block = MLPBlock(
            d_model=d_model,
            d_ffn=d_ffn,
            dropout_prob=dropout_prob,
            use_mlp_bias=use_mlp_bias,
            rms_norm_eps=rms_norm_eps,
            device=device,
            dtype=dtype
        )

    def forward(
        self, 
        x: Tensor,
        use_qk_norm: bool = True,
        use_mqa: bool = False,
        qk_norm_eps: float = 1e-10,
        qk_norm_type: int = 2,
        use_cache: bool = False,
        layer_idx: Optional[int] = None,
        kv_cache: Optional[SpatialKVCache] = None,
        is_causal: bool = False,
        padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass of spatial transformer block.
        
        Args:
            x (Tensor): Input tensor of shape [B, H*W, d_model].
            use_qk_norm (bool): Whether to use QK normalization or not.
            use_mqa (bool): Whether to use MQA or not.
            qk_norm_eps (float): Epsilon value for QK normalization.
            qk_norm_type (int): Type of normalization.
            use_cache (bool): Whether to use cache or not.
            layer_idx (Optional[int]): Layer index for KV cache.
            kv_cache (Optional[SpatialKVCache]): KV caching module.
            is_causal (bool): Whether to use causal masking or not.
            padding_mask (Optional[Tensor]): Padding tensor of shape [B, H*W].
        """
        with autocast(device_type=x.device.type):
            return self.mlp_block(
                self.attn_block(
                    x,
                    use_qk_norm=use_qk_norm,
                    use_mqa=use_mqa,
                    qk_norm_eps=qk_norm_eps,
                    qk_norm_type=qk_norm_type,
                    use_cache=use_cache,
                    layer_idx=layer_idx,
                    kv_cache=kv_cache,
                    is_causal=is_causal,
                    padding_mask=padding_mask
                )
            )
