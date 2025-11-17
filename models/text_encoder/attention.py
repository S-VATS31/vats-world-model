from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast

from utils.attention import extend_kv_heads, apply_qk_norm

class RoPE(nn.Module):
    """RoPE module for text encoder.
    
    Args:
        head_dim (int): Dimension of each attention head.
        rope_theta (float): Exponential base of inverse frequency.
        device (torch.device): Accelerator at use.
        dtype (torch.dtype): Data type of tensors.
    """
    def __init__(
        self,
        head_dim: int,
        rope_theta: float,
        device: torch.device,
        dtype: torch.dtype
    ):
        super().__init__()

        self.head_dim = head_dim
        self.rope_theta = rope_theta

        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be divisible by 2, got {head_dim}")

        inv_freq = 1.0 / (
            rope_theta ** (
                torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim
            )
        )

        # use persistent=False to avoid state_dict errors
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Applies rotary position embeddings to input tensor.

        Args:
            x (Tensor): Input tensor of shape [B, T, num_heads, head_dim].

        Returns:
            Tensor with RoPE applied, same shape as input.
        """
        T = x.size(1)

        pos = torch.arange(T, device=x.device, dtype=x.dtype)
        freqs = torch.einsum("i,j->ij", pos, self.inv_freq)
        embedding = torch.cat([freqs, freqs], dim=-1) # [T, head_dim]

        cos_embed = embedding.cos()[None, :, None, :]
        sin_embed = embedding.sin()[None, :, None, :]

        x_even, x_odd = x[..., ::2], x[..., 1::2] # 2i for even, 2i+1 for odd
        rotated = torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)

        return x * cos_embed + rotated * sin_embed # [B, T, num_heads, head_dim]


class Attention(nn.Module):
    """Attention module for encoder.
    
    Args:
        num_heads (int): Number of attention heads.
        d_model (int): Dimensionality of model embeddings.
        query_groups (int): Number of query groups.
        rope_theta (float): Exponential base of inverse frequency.
        softmax_scale (float): Softmax scale for attention computatation.
        use_proj_bias (bool): Whether to use projection bias or not.
        use_qkv_proj (bool): Whether to use QKV projection or not.
        device (torch.device): Accelerator at use.
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
        device: torch.device,
        dtype: torch.dtype
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model must be divisble by num_heads, got {d_model} % {num_heads} != 0."
            )
        if num_heads % query_groups != 0:
            raise ValueError(
                f"num_heads must be divisble by query_groups, got {num_heads} % {query_groups} != 0."
            )

        self.num_heads = num_heads
        self.d_model = d_model
        self.query_groups = query_groups
        self.rope_theta = rope_theta
        self.softmax_scale = softmax_scale
        self.use_proj_bias = use_proj_bias
        self.use_qkv_proj = use_qkv_proj
        self.device = device
        self.dtype = dtype
        self.head_dim = d_model // num_heads
        self.heads_per_group = self.num_heads // self.query_groups

        # Setup projections
        if use_qkv_proj:
            self.w_qkv = nn.Linear(
                d_model,
                num_heads*self.head_dim + 2*self.query_groups*self.head_dim,
                use_proj_bias,
                device,
                dtype
            )
        else:
            self.w_q = nn.Linear(d_model, d_model, use_proj_bias, device, dtype)
            self.w_k = nn.Linear(d_model, query_groups*self.head_dim, use_proj_bias, device, dtype)
            self.w_v = nn.Linear(d_model, query_groups*self.head_dim, use_proj_bias, device, dtype)

        self.w_o = nn.Linear(d_model, d_model, use_proj_bias, device, dtype)

        # RoPE
        self.rope = RoPE(self.head_dim, rope_theta, device, dtype)

    def _setup_qkv(
        self, 
        x: Tensor, 
        use_qk_norm: bool, 
        use_mqa: bool,
        eps: Optional[float] = 1e-10,
        norm: Optional[int] = 2
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Setup query, key, and value tensors.
        
        Args:
            x (Tensor): Input tensor of shape [B, T, d_model].
            use_qk_norm (bool): Whether to use QK normalization or not.
            use_mqa (bool): Whether to use MQA or not.
            eps (Optional[float]): Small float to maintain numerical stability in QK normalization. Defaults to 1e-10.
            norm (Optional[int]): Type of normalization to use. Defaults to L2 normalization.

        Returns:
            Tuple:
                Tensor: Query tensor of shape [B, T, num_heads, head_dim].
                Tensor: Key tensor of shape [B, T, num_heads or 1, head_dim].
                Tensor: Value tensor of shape [B, T, num_heads or 1, head_dim].
        """
        B, T, _ = x.shape

        # Handle 0 token input
        if T == 0:
            return (
                torch.empty(B, 0, self.num_heads, self.head_dim),
                torch.empty(B, 0, self.num_heads, self.head_dim),
                torch.empty(B, 0, self.num_heads, self.head_dim)
            )
        if self.use_qkv_proj:
            qkv = self.w_qkv(x)
            # q: [B, T, num_heads * head_dim]
            q, kv = torch.split(
                qkv, [self.num_heads * self.head_dim, 2 * self.query_groups * self.head_dim], dim=-1
            )
            # k, v: [B, T, query_groups * head_dim]
            k, v = kv.chunk(chunks=2, dim=-1)
        else:
            q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        # q: [B, T, num_heads, head_dim]; k, v: [B, T, query_groups, head_dim]
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.query_groups, self.head_dim)
        v = v.view(B, T, self.query_groups, self.head_dim)

        # QK norm
        if use_qk_norm:
            q = apply_qk_norm(q, eps=eps, norm=norm)
            k = apply_qk_norm(k, eps=eps, norm=norm)

        # RoPE
        q = self.rope(q)
        k = self.rope(k)

        # Extend KV heads
        k = extend_kv_heads(k, dim=2, repeats=self.heads_per_group, use_mqa=use_mqa)
        v = extend_kv_heads(v, dim=2, repeats=self.heads_per_group, use_mqa=use_mqa)

        # Permute to [B, num_heads, T, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        return q, k, v

    def _scaled_dot_product_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Scaled dot product attention using PyTorch SDPA.
        
        Args:
            query (Tensor): Query tensor of shape [B, num_heads, T, head_dim].
            key (Tensor): Key tensor of shape [B, num_heads or 1, T, head_dim].
            value (Tensor): Value tensor of shape [B, num_heads or 1, T, head_dim].
            padding_mask (Tensor): Padding tensor of shape [B, T_q].

        Returns:
            Tensor: Output tensor of shape [B, T, d_model].
        """
        B, _, T_q, _= query.shape
        T_k = key.size(2)
        
        # Handle 0 token input
        if query.size(1) == 0 or key.size(1) == 0:
            return torch.empty(B, 0, self.d_model)
        
        # Handle padding tensor
        if padding_mask is not None:
            padding_mask = padding_mask.bool()
            padding_mask = padding_mask[:, None, :, None] # [B, 1, T_q, 1]
            attn_mask = padding_mask.expand(B, 1, T_q, T_k) # [B, 1, T_q, T_k]
        else:
            attn_mask = None

        attn_out = F.scaled_dot_product_attention(
            query, key, value, attn_mask, scale=self.softmax_scale
        ) # [B, num_heads, T, head_dim]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_q, -1)
        
        return self.w_o(attn_out)

    def forward(
        self,
        x: torch.Tensor,
        use_qk_norm: bool,
        use_mqa: bool,
        eps: Optional[float] = 1e-10,
        norm: Optional[int] = 2,
        padding_mask: Optional[Tensor] = None,
        *,
        _return_qkv: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Forward pass of attention module.
        
        Args:
            x (Tensor): Input tensor of shape [B, T, d_model].
            use_qk_norm (bool): Whether to use QK normalization or not.
            use_mqa (bool): Whether to use MQA or not.
            eps (Optional[float]): Small value to maintain numerical stability in QK normalization.
            norm (Optional[int]): Type of normalization for QK normalization.
            padding_mask (Optional[Tensor]): Padding tensor of shape [B, T].
        """
        with autocast(device_type=x.device.type, dtype=x.dtype):
            q, k, v = self._setup_qkv(x, use_qk_norm, use_mqa, eps, norm)
            if _return_qkv:
                return (self._scaled_dot_product_attention(q, k, v, padding_mask), q, k, v)
            return self._scaled_dot_product_attention(q, k, v, padding_mask)


class AttentionBlock(nn.Module):
    """Attention block for attention, normalization, residuals, and dropout.
    
    Args:
        num_heads (int): Number of attention heads.
        d_model (int): Dimensionality of model embeddings.
        query_groups (int): Number of query groups.
        rope_theta (float): Exponential base for inverse frequency.
        softmax_scale (float): Softmax scale for attention computation.
        use_proj_bias (bool): Whether to use projection bias or not.
        use_qkv_proj (bool): Whether to use QKV projection or not.
        rms_norm_eps (float): Epsilon value for RMSNorm.
        dropout_prob (float): Dropout probability for training.
        device (torch.device): Accelerator at use.
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
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()

        self.attn = Attention(
            num_heads=num_heads,
            d_model=d_model,
            query_groups=query_groups,
            rope_theta=rope_theta,
            softmax_scale=softmax_scale,
            use_proj_bias=use_proj_bias,
            use_qkv_proj=use_qkv_proj,
            device=device,
            dtype=dtype
        )
        self.rms_norm = nn.RMSNorm(
            d_model, 
            eps=rms_norm_eps, 
            device=device, 
            dtype=dtype
        )
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(
        self,
        x: Tensor,
        use_qk_norm: bool,
        use_mqa: bool,
        eps: Optional[float] = 1e-10,
        norm: Optional[int] = 2,
        padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass of the attention block.
        
        Args:
            x (Tensor): Input tensor of shape [B, T, d_model].
            use_qk_norm (bool): Whether to use QK normalization or not.
            use_mqa (bool): Whether to use MQA or not.
            eps (Optional[float]): Epsilon value for QK normalization.
            norm (Optoinal[float]): Normalization type for QK tensors.
            padding_mask (Optional[float]): Padding tensor of shape [B, T].

        Returns:
            Tensor: Output tensor of same shape as input.
        """
        with autocast(device_type=x.device.type, dtype=x.dtype):
            return x + self.dropout(
                self.attn(
                    self.rms_norm(x),
                    use_qk_norm=use_qk_norm,
                    use_mqa=use_mqa,
                    eps=eps,
                    norm=norm,
                    padding_mask=padding_mask
                )
            )
        