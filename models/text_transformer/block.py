from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast

from models.mlp import MLPBlock
from models.text_transformer.kv_cache import KVCache
from models.text_transformer.attention import CausalAttentionBlock

class CausalTransformerBlock(nn.Module):
    """Causal transformer block.
    
    Args:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads.
        query_groups (int): Number of query groups for GQA.
        use_qkv_bias (bool): Whether to use QKV bias or not.
        use_o_bias (bool): Whether to use output bias or not.
        use_qkv_proj (bool): Whether to use full QKV projection or not.
        rope_theta (float): Exponential base for RoPE.
        softmax_scale (float): Scaler for attention computation.
        rms_norm_eps (float): Epsilon value for RMSNorm.
        dropout_prob (float): Dropout probability for regularization.
        d_ffn (int): Dimensionality of MLP.
        use_mlp_bias (bool): Whether to us MLP bias or not.
        device (torch.device): Accelerator at use.
        dtype (torch.dtype): Data type of model parameters.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        use_qkv_bias: bool,
        use_o_bias: bool,
        use_qkv_proj: bool,
        rope_theta: float,
        softmax_scale: float,
        rms_norm_eps: float,
        dropout_prob: float,
        d_ffn: int,
        use_mlp_bias: bool,
        device: torch.device,
        dtype: torch.dtype
    ):
        super().__init__()
        
        self.attn_block = CausalAttentionBlock(
            d_model=d_model,
            num_heads=num_heads,
            query_groups=query_groups,
            use_qkv_bias=use_qkv_bias,
            use_o_bias=use_o_bias,
            use_qkv_proj=use_qkv_proj,
            rope_theta=rope_theta,
            softmax_scale=softmax_scale,
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
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None,
        use_cache: bool = False,
        is_causal: bool = True,
        padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass of the causal attention block.

        Args:
            x (Tensor): Input tensor of shape [B, T, d_model].
            use_qk_norm (bool): Whether to apply QK normalization
            use_mqa (bool): Whether to use MQA.
            qk_norm_eps (float): Epsilon value for QK normalization.
            qk_norm_type (int): Type of QK normalization to apply.
            kv_cache (Optional[KVCache]): KV caching module.
            layer_idx (Optional[int]): Layer index to update KVs with respect to.
            use_cache (bool): Whether to use KV caching or not.
            is_causal (bool): Whether to use causal masking or not.
            padding_mask (Optional[Tensor]): Padding tensor of shape [B, T].

        Returns:
            Tensor: Output tensor of same shape as input.
        """
        with autocast(device_type=x.device.type):
            return self.mlp_block(
                self.attn_block(
                    x,
                    use_qk_norm=use_qk_norm,
                    use_mqa=use_mqa,
                    qk_norm_eps=qk_norm_eps,
                    qk_norm_type=qk_norm_type,
                    kv_cache=kv_cache,
                    layer_idx=layer_idx,
                    use_cache=use_cache,
                    is_causal=is_causal,
                    padding_mask=padding_mask
                )
            )
