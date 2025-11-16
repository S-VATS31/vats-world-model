from typing import *

import torch
from torch import Tensor
import torch.nn as nn
from torch.amp import autocast

from models.text_encoder.attention import AttentionBlock
from models.text_encoder.mlp import MLPBlock

class TransformerBlock(nn.Module):
    """Block applying transformations of attention and MLP block.

    Args:
        num_heads (int): Number of attention heads.
        d_model (int): Dimensionality of model embeddings.
        query_groups (int): Number of query groups for GQA.
        rope_theta (float): Exponential base of inverse frequency.
        softmax_scale (float): Softmax scale for attention computation.
        use_proj_bias (bool): Whether to use projection bias or not.
        use_qkv_proj (bool): Whether to use QKV projection or not.
        rms_norm_eps (float): RMSNorm epsilon value.
        dropout_prob (float): Dropout probability for regularization.
        d_ffn (int): Dimensionality of MLP.
        use_mlp_bias (bool): Whether to use MLP bias or not.
        device (torch.device): Accelerator.
        dtype (torch.dtype): Data type of tensors.
    """
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        query_groups: int,
        rope_theta: float,
        softmax_scale: float,
        use_proj_bias: bool,
        use_qkv_proj: bool,
        rms_norm_eps: float,
        dropout_prob: float,
        d_ffn: int,
        use_mlp_bias: bool,
        device: torch.device,
        dtype: torch.dtype
    ):
        super().__init__()

        self.attn_block = AttentionBlock(
            num_heads=num_heads,
            d_model=d_model,
            query_groups=query_groups,
            rope_theta=rope_theta,
            softmax_scale=softmax_scale,
            use_proj_bias=use_proj_bias,
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
        use_qk_norm: bool,
        use_mqa: bool,
        eps: Optional[float] = 1e-10,
        norm: Optional[int] = 2,
        padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass of transformer block.
        
        Args:
            x (Tensor): Input tensor of shape [B, T, d_model].
            use_qk_norm (bool): Whether to use QK normalization or not.
            use_mqa (bool): Whether to use MQA or not.
            eps (float): Epsilon value for QK normalization.
            norm (int): Type of normalization for QK normalization.
            padding_mask (Tensor): Padding tensor of shape [B, T].

        Returns:
            Tensor: Output tensor of same shape.
        """
        with autocast(device_type=x.device.type, dtype=x.dtype):
            return self.mlp_block(
                self.attn_block(
                    x,
                    use_qk_norm=use_qk_norm,
                    use_mqa=use_mqa,
                    eps=eps,
                    norm=norm,
                    padding_mask=padding_mask
                )
            )
