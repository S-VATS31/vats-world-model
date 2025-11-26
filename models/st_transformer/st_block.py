from typing import Optional, Literal

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast

from models.mlp import MLPBlock
from models.st_transformer.temporal_cache import TemporalKVCache
from models.st_transformer.st_attention import SpatioTemporalAttentionBlock

class SpatioTemporalTransformerBlock(nn.Module):
    """Spatiotemporal transformer block.
    
    Args:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads.
        query_groups (int): Number of query groups for GQA.
        rope_theta (float): Exponential base for RoPE.
        softmax_scale (float): Softmax scale for attention computation.
        use_qkv_bias (bool): Whether to use QKV bias or not.
        use_o_bias (bool): Whether to use output bias or not.
        use_qkv_proj (bool): Whether to use QKV projection or not.
        rms_norm_eps (float): Epsilon value for RMSNorm.
        dropout_prob (float): Dropout probability for regularization.
        d_ffn (int): Dimensionality of MLP.
        use_mlp_bias (bool): Whether to use bias in MLP projections.
        device (torch.device): Accelerator at use.
        dtype (torch.dtype): Data type of tensors.
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
        
        self.attn_block = SpatioTemporalAttentionBlock(
            num_heads=num_heads,
            d_model=d_model,
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
        H_patch: int,
        W_patch: int,
        T_patch: int,
        use_qk_norm: bool = True,
        use_mqa: bool = False,
        qk_norm_eps: float = 1e-10,
        qk_norm_type: int = 2,
        is_causal: bool = True,
        use_cache: bool = False,
        kv_cache: Optional[TemporalKVCache] = None,
        layer_idx: Optional[int] = None,
        padding_mask: Optional[Tensor] = None,
        attention_interleave: Literal["st", "ts"] = "st"
    ) -> Tensor:
        """Forward pass of ST transformer block.
        
        Args:
            x (Tensor): Input tensor of shape [B, T_frames, H*W, d_model].
            H_patch (int): Height of each patch.
            W_patch (int): Width of each patch.
            T_patch (int): Temporal length of each patch.
            use_qk_norm (bool): Whether to use QK normalization or not.
            use_mqa (bool): Whether to use MQA or not.
            qk_norm_eps (float): Epsilon value for QK norm.
            qk_norm_type (int): Type of normalization for QK norm.
            is_causal (bool): Whether to use causal masking or not.
            use_cache (bool): Whether to use cache or not.
            kv_cache (Optional[TemporalKVCache]): KV caching module.
            layer_idx (Optional[int]): Layer index to update KVs.
            padding_mask (Optional[Tensor]): Padding tensor of shape [B, T_frames].
            attention_interleave (Literal["st", "ts"]): Interleaving method.
                st: Sequential application as TA(SA(x)).
                ts: Sequential application as SA(TA(x)).

        Returns:
            Tensor: Output tensor of shape [B, T_frames, H*W, d_model].
        """
        with autocast(device_type=x.device.type):
            return self.mlp_block(
                self.attn_block(
                    x,
                    H_patch=H_patch,
                    W_patch=W_patch,
                    T_patch=T_patch,
                    use_qk_norm=use_qk_norm,
                    use_mqa=use_mqa,
                    qk_norm_eps=qk_norm_eps,
                    qk_norm_type=qk_norm_type,
                    is_causal=is_causal,
                    use_cache=use_cache,
                    kv_cache=kv_cache,
                    layer_idx=layer_idx,
                    padding_mask=padding_mask,
                    attention_interleave=attention_interleave
                )
            )
