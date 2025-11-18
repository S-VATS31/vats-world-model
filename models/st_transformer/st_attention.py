from typing import Union, Tuple, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast

from utils.attention import apply_qk_norm, extend_kv_heads
from models.st_transformer.temporal_cache import TemporalKVCache

class RoPE3D(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x: Tensor) -> Tensor:
        with autocast(device_type=x.device.type):
            pass


class SpatioTemporalAttention(nn.Module):
    """Factorized attention module.
    
    Args:
        num_heads (int): Number of attention heads.
        d_model (int): Dimensionality of model embeddings.
        query_groups (int): Number of query groups for GQA.
        rope_theta (float): Exponential base for RoPE.
        softmax_scale (float): Scaler for attention computation.
        use_qkv_bias (bool): Whether to use bias in QKV projections.
        use_o_bias (bool): Whether to use bias in O projection.
        use_qkv_proj (bool): Whether to use QKV projection.
        max_temporal_len (int): Max input frames for RoPE.
        max_spatial_len (int): Max height and width of frames for RoPE..
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
        use_qkv_bias: bool,
        use_o_bias: bool,
        use_qkv_proj: bool,
        max_temporal_len: int, # keep if RoPE uses
        max_spatial_len: int,  # keep if RoPE uses
        device: torch.device,
        dtype: torch.dtype
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model must be divisble by num_heads, got {d_model} % {num_heads} != 0"
            )
        if num_heads % query_groups != 0:
            raise ValueError(
                f"num_heads must be divisible by query_groups, got {num_heads} % {query_groups} != 0"
            )
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.query_groups = query_groups
        self.softmax_scale = softmax_scale
        self.use_qkv_proj = use_qkv_proj
        self.device = device
        self.dtype = dtype
        self.head_dim = self.d_model // self.num_heads
        self.heads_per_group = self.num_heads // self.query_groups

        # Set up QKV projections
        if use_qkv_proj:
            self.w_qkv = nn.Linear(
                d_model,
                num_heads*self.head_dim + 2*self.query_groups*self.head_dim,
                bias=use_qkv_bias,
                device=device,
                dtype=dtype
            )
        else:
            self.w_q = nn.Linear(
                d_model, d_model, bias=use_qkv_bias, device=device, dtype=dtype
            )
            self.w_k = nn.Linear(
                d_model, 
                query_groups*self.head_dim, 
                bias=use_qkv_bias, 
                device=device, 
                dtype=dtype
            )
            self.w_v = nn.Linear(
                d_model, 
                query_groups*self.head_dim, 
                bias=use_qkv_bias, 
                device=device,
                dtype=dtype
            )

        # Set up output projection
        self.w_o = nn.Linear(d_model, d_model, bias=use_o_bias, device=device, dtype=dtype)

        # Set up RoPE
        # TODO: add RoPE implementation and instantiate here

    def _setup_spatial_qkv(
        self,
        x: Tensor,
        use_qk_norm: bool = True,
        use_mqa: bool = False,
        qk_norm_eps: float = 1e-10,
        qk_norm_type: int = 2
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Set up spatial QKV tensors.
        
        Args:
            x (Tensor): Input tensor of shape [B, T_frames, H*W, d_model].
            use_qk_norm (bool): Whether to use QK normalization.
            use_mqa (bool): Whether to use MQA or not.
            qk_norm_eps (float): Epsilon value for QK norm.
            qk_norm_type (int): Type of normalization for QK norm.

        Returns:
            Tuple:
                Tensor: Query tensor of shape [B*T_frames, num_heads, H*W, head_dim].
                Tensor: Key tensor of shape [B*T_frames, num_heads or 1, H*W, head_dim].
                Tensor: Value tensor of shape [B*T_frames, num_heads, H*W head_dim].
        """
        B, T_frames, num_spatial_patches, _ = x.shape

        # Handle zero input frames
        if T_frames == 0:
            return (
                torch.empty(
                    B*T_frames, self.num_heads, num_spatial_patches, self.head_dim, 
                    device=x.device, dtype=x.dtype
                ),
                torch.empty(
                    B*T_frames, self.num_heads, num_spatial_patches, self.head_dim, 
                    device=x.device, dtype=x.dtype
                ),
                torch.empty(
                    B*T_frames, self.num_heads, num_spatial_patches, self.head_dim, 
                    device=x.device, dtype=x.dtype
                )
            )
        
        # Reshape input tensor for spatial QKV
        x = x.view(-1, num_spatial_patches, self.d_model) # [B*T_frames, H*W, d_model]

        # Get QKV tensors
        if self.use_qkv_proj:
            qkv = self.w_qkv(x)
            # q: [B*T_frames, H*W, num_heads*head_dim] 
            # kv: [B*T_frames, H*W, 2*query_groups*head_dim]
            q, kv = torch.split(
                qkv, 
                [self.num_heads * self.head_dim, 2 * self.query_groups * self.head_dim], 
                dim=-1
            )
            # k, v: [B*T_frames, H*W, query_groups*head_dim]
            k, v = kv.chunk(chunks=2, dim=-1)
        else:
            q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        # Reshape QKV
        q = q.view(-1, num_spatial_patches, self.num_heads, self.head_dim)
        k = k.view(-1, num_spatial_patches, self.query_groups, self.head_dim)
        v = v.view(-1, num_spatial_patches, self.query_groups, self.head_dim)

        # Apply QK norm
        if use_qk_norm:
            q = apply_qk_norm(q, eps=qk_norm_eps, norm=qk_norm_type)
            k = apply_qk_norm(k, eps=qk_norm_eps, norm=qk_norm_type)

        # Apply RoPE
        # TODO: Apply forward pass of RoPE to qk tensors

        # Extend KV heads
        k = extend_kv_heads(k, dim=2, repeats=self.heads_per_group, use_mqa=use_mqa)
        v = extend_kv_heads(v, dim=2, repeats=self.heads_per_group, use_mqa=use_mqa)

        # q: [B*T_frames, num_heads, H*W, head_dim]
        # k, v: [B*T_frames, num_heads or 1, H*W, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        return q, k, v

    def _setup_temporal_qkv(
        self,
        x: Tensor,
        use_qk_norm: bool = True,
        use_mqa: bool = False,
        qk_norm_eps: float = 1e-10,
        qk_norm_type: int = 2,
        use_cache: bool = False,
        kv_cache: Optional[TemporalKVCache] = None,
        layer_idx: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Setup temporal QKV tensors.
        
        Args:
            x (Tensor): Input tensor of shape of [B, T_frames, H*W, d_model].
            use_qk_norm (bool): Whether to use QK normalization or not.
            use_mqa (bool): Whether to use MQA or not.
            qk_norm_eps (float): QK norm epsilon value.
            qk_norm_type (int): L_(qk_norm_type) normalization.
            use_cache (bool): Whether to use KV cache.
            kv_cache (Optional[TemporalKVCache]): KV caching module.
            layer_idx (Optional[int]): Layer to update KV's with respect to.

        Returns:
            Tuple:
                Tensor: Query tensor of shape [B*H*W, num_heads, T_frames, head_dim].
                Tensor: Key tensor of shape [B*H*W, num_heads or 1, T_frames_total, head_dim].
                Tensor: Value tensor of shape [B*H*W, num_heads or 1, T_frames_total, head_dim].
        """
        B, T_frames, num_spatial_patches, _ = x.shape

        # Handle zero frames input
        if T_frames == 0:
            return (
                torch.empty(
                    B*num_spatial_patches, self.num_heads, T_frames, self.head_dim,
                    device=x.device, dtype=x.dtype
                ),
                torch.empty(
                    B*num_spatial_patches, self.num_heads, T_frames, self.head_dim,
                    device=x.device, dtype=x.dtype
                ),
                torch.empty(
                    B*num_spatial_patches, self.num_heads, T_frames, self.head_dim,
                    device=x.device, dtype=x.dtype
                )
            )

        # x: [B*H*W, T_frames, d_model]
        x = x.view(-1, T_frames, self.d_model)

        # Get QKV tensors
        if self.use_qkv_proj:
            qkv = self.w_qkv(x)
            # q: [B*H*W, T_frames, num_heads*head_dim]
            # k, v: [B*H*W, T_frames, 2*query_groups*head_dim]
            q, kv = torch.split(
                qkv,
                [self.num_heads*self.head_dim, 2*self.query_groups*self.head_dim],
                dim=-1
            )
            # k, v: [B*H*W, T_frames, query_groups*head_dim]
            k, v = kv.chunk(chunks=2, dim=-1)
        else:
            q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        # Reshape QKV tensors
        q = q.view(B*num_spatial_patches, T_frames, self.num_heads, self.head_dim)
        k = k.view(B*num_spatial_patches, T_frames, self.query_groups, self.head_dim)
        v = v.view(B*num_spatial_patches, T_frames, self.query_groups, self.head_dim)

        # Apply QK norm
        if use_qk_norm:
            q = apply_qk_norm(q, eps=qk_norm_eps, norm=qk_norm_type)
            k = apply_qk_norm(k, eps=qk_norm_eps, norm=qk_norm_type)

        # Apply RoPE
        # TODO: apply RoPE forward pass here

        # Extend KV heads
        k = extend_kv_heads(k, dim=2, repeats=self.heads_per_group, use_mqa=use_mqa)
        v = extend_kv_heads(v, dim=2, repeats=self.heads_per_group, use_mqa=use_mqa)

        # Handle KV caching
        if use_cache and kv_cache is not None and layer_idx is not None:
            # Permute to match cache format: [B*H*W, num_heads, T_frames, head_dim]
            k_new = k.permute(0, 2, 1, 3)  # [B*H*W, num_heads, T_frames, head_dim]
            v_new = v.permute(0, 2, 1, 3)  # [B*H*W, num_heads, T_frames, head_dim]
            
            # Get ALL previously cached frames (not just T_frames)
            past_k, past_v = kv_cache.get_cached_kv(layer_idx)
            
            # Update cache with ONLY the new k/v
            kv_cache.update(layer_idx, k_new, v_new)
            
            # Concatenate past with current for attention computation
            if past_k is not None and past_v is not None:
                k_new = torch.cat([past_k, k_new], dim=2)  # [B*H*W, num_heads, T_total, head_dim]
                v_new = torch.cat([past_v, v_new], dim=2)  # [B*H*W, num_heads, T_total, head_dim]
            
            # Use the concatenated k,v for attention (already permuted)
            return q.permute(0, 2, 1, 3), k_new, v_new
        
        # q: [B*H*W, num_heads, T_frames, head_dim]
        # k, v: [B*H*W, num_heads or 1, T_frames, head_dim]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        return q, k, v

    def _spatial_attention(
        self,
        x: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Spatial attention applied to input tensor.
        
        Args:
            x (Tensor): Input tensor of shape [B, T_frames, H*W, d_model].
            query (Tensor): Query tensor of shape [B*T_frames, num_heads, H*W, head_dim].
            key (Tensor): Key tensor of shape [B*T_frames, num_heads or 1, H*W, head_dim].
            value (Tensor): Value tensor of shape [B*T_frames, num_heads or 1, H*W, head_dim].
            padding_mask (Optional[Tensor]): Padding tensor of shape [B, T_frames].

        Returns:
            Tensor: Output tensor of shape [B, T, H*W, d_model].
        """
        B, T_frames, num_spatial_patches, _ = x.shape

        # Handle zero input tokens
        if query.size(0) == 0 or key.size(0) == 0:
            return torch.empty_like(x, device=x.device, dtype=x.dtype)
        
        # Handle padding mask
        if padding_mask is not None:
            padding_mask_bool = padding_mask.bool()
            padding_mask = padding_mask_bool.view(-1).unsqueeze(-1) # [B*T_frames, 1]
            padding_mask = padding_mask.expand(B*T_frames, num_spatial_patches)
            query_padding_mask = padding_mask[:, None, :, None] # [B*T, 1, H*W, 1]
            key_padding_mask = padding_mask[:, None, None, :] # [B*T, 1, 1, H*W]
            attn_mask = torch.logical_or(query_padding_mask, key_padding_mask)
        else:
            attn_mask = None

        # Compute attention output
        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask
        ) # [B*T, num_heads, H*W, head_dim]

        # Reshape output back to [B, T, H*W, d_model]
        out = out.transpose(1, 2).contiguous().view(
            B, T_frames, num_spatial_patches, self.d_model
        )

        return self.w_o(out)

    def _temporal_attention(
        self,
        x: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        is_causal: bool = True,
        padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Temporal attention applied to input tensor.
        
        Args:
            x (Tensor): Input tensor of shape [B, T_frames, H*W, d_model].
            query (Tensor) Query tensor of shape [B*H*W, num_heads, T_frames, head_dim].
            key (Tensor): Key tensor of shape [B*H*W, num_heads or 1, T_frames, head_dim].
            value (Tensor): Value tensor of shape [B*H*W, num_heads or 1, T_frames, head_dim].
            is_causal (bool): Whether to use causal masking or not. Defaults to True.
            padding_mask (Optional[Tensor]): Padding tensor of shape [B, T_frames].
        
        Returns:
            Tensor: Output tensor of shape [B, T, H*W, d_model].
        """
        B, T_frames, num_spatial_patches, _ = x.shape

        # Handle zero input frames
        if query.size(2) == 0 or key.size(2) == 0:
            return torch.empty_like(x, device=x.device, dtype=x.dtype)
        
        # Handle padding mask
        if padding_mask is not None:
            # [B, 1, T_frames]
            padding_mask = padding_mask.bool().unsqueeze(1).expand(B, num_spatial_patches, T_frames)
            padding_mask = padding_mask.contiguous().reshape(-1, T_frames)
            query_padding_mask = padding_mask[:, None, :, None] # [B, 1, T_frames, 1]
            key_padding_mask = padding_mask[:, None, None, :] # [B, 1, 1, T_frames]
            attn_mask = torch.logical_or(query_padding_mask, key_padding_mask)
            
            # Handle temporal causal masking
            if is_causal:
                # causal_mask: [T_frames, T_frames], True = pad, False = compute attention
                causal_mask = torch.triu(
                    torch.ones(T_frames, T_frames, device=x.device, dtype=torch.bool),
                    diagonal=1
                )
                causal_mask = causal_mask[None, None, :, :] # [1, 1, T_frames, T_frames]
                attn_mask = torch.logical_or(attn_mask, causal_mask)
        else:
            # No padding, check for causal
            if is_causal:
                causal_mask = torch.triu(
                    torch.ones(T_frames, T_frames, device=x.device, dtype=torch.bool),
                    diagonal=1
                )
                # Only constraint is causal masking
                attn_mask = causal_mask[None, None, :, :] # [1, 1, T_frames, T_frames]
            else:
                attn_mask = None

        # Compute attention output
        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask
        ) # [B*H*W, num_heads, T_frames, head_dim]

        # out: [B, T_frames, H*W, d_model]
        out = out.transpose(1, 2).contiguous().view(
            B, T_frames, num_spatial_patches, self.d_model
        )

        return self.w_o(out)

    def forward(
        self, 
        x: Tensor,
        use_qk_norm: bool = True,
        use_mqa: bool = False,
        qk_norm_eps: float = 1e-10,
        qk_norm_type: int = 2,
        is_causal: bool = True,
        use_cache: bool = False,
        kv_cache: Optional[TemporalKVCache] = None,
        layer_idx: Optional[int] = None,
        padding_mask: Optional[Tensor] = None,
        attention_interleave: Literal["st", "ts"] = "st",
        *,
        _return_debug: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        """Forward pass of ST transformer.
        
        Args:
            x (Tensor): Input tensor of shape [B, T_frames, H*W, d_model].
            use_qk_norm (bool): Whether to use QK normalization or not.
            use_mqa (bool): Whether to use MQA or not.
            qk_norm_eps (float): Epsilon value for QK norm.
            qk_norm_type (int): Type of normalization for QK norm.
            is_causal (bool): Whether to use causal masking or not.
            use_cache (bool): Whether to use cache or not.
            kv_cache (Optional[TemporalKVCache]): KV caching module.
            layer_idx (Optional[int]): Layer index to update KVs.
            padding_mask (Optional[Tensor]): Padding tensor of shape [B, T_frames.]
            attention_interleave (Literal["st", "ts"]): Interleaving method.
                st: Sequential application as TA(SA(x)).
                ts: Sequential application as SA(TA(x)).
        
        Kwargs:
            _return_debug (bool): Return QKV tensors for debugging.

        Returns:
            Union[Tensor, Tuple[Tensor, ...]]:
                Tensor: Attention output.
                Tuple[Tensor, ...]: Attention output + QKV tensors.
        """
        with autocast(device_type=x.device.type):
            # Get temporal QKV
            query_temporal, key_temporal, value_temporal = self._setup_temporal_qkv(
                x,
                use_qk_norm=use_qk_norm,
                use_mqa=use_mqa,
                qk_norm_eps=qk_norm_eps,
                qk_norm_type=qk_norm_type,
                use_cache=use_cache,
                kv_cache=kv_cache,
                layer_idx=layer_idx
            )
            # Get spatial QKV
            query_spatial, key_spatial, value_spatial = self._setup_spatial_qkv(
                x,
                use_qk_norm=use_qk_norm,
                use_mqa=use_mqa,
                qk_norm_eps=qk_norm_eps,
                qk_norm_type=qk_norm_type
            )
            if attention_interleave == "st":
                # spatial_out: [B, T_frames, H*W, d_model]
                spatial_out = self._spatial_attention(
                    x,
                    query=query_spatial,
                    key=key_spatial,
                    value=value_spatial,
                    padding_mask=padding_mask
                )
                # out: [B, T_frames, H*W, d_model]
                out = self._temporal_attention(
                    spatial_out, 
                    query=query_temporal, 
                    key=key_temporal, 
                    value=value_temporal, 
                    is_causal=is_causal, 
                    padding_mask=padding_mask
                )
            elif attention_interleave == "ts":
                # temporal_out: [B, T_frames, H*W, d_model]
                temporal_out = self._temporal_attention(
                    x, 
                    query=query_temporal, 
                    key=key_temporal, 
                    value=value_temporal, 
                    is_causal=is_causal, 
                    padding_mask=padding_mask
                )
                # out: [B, T_frames, H*W, d_model]
                out = self._spatial_attention(
                    temporal_out,
                    query=query_spatial,
                    key=key_spatial,
                    value=value_spatial,
                    padding_mask=padding_mask
                )
            else:
                raise ValueError(
                    f"expected 'st' or 'ts', got {attention_interleave}"
                )
            # Return debugging values
            if _return_debug:
                return (
                    out,
                    query_spatial,
                    key_spatial,
                    value_spatial,
                    query_temporal,
                    key_temporal,
                    value_temporal
                )
            return out


class SpatioTemporalAttentionBlock(nn.Module):
    """SpatioTemporal attention block.
    
    Args:
        num_heads (int): Number of attention heads.
        d_model (int): Dimensionality of model embeddings.
        query_groups (int): Number of query groups for GQA.
        rope_theta (float): Exponential base for RoPE.
        softmax_scale (float): Softmax scale for attention computation.
        use_qkv_bias (bool): Whether to use QKV bias or not.
        use_o_bias (bool): Whether to use output bias or not.
        use_qkv_proj (bool): Whether to use QKV projection or not.
        rms_norm_eps (float): Epsilon value for RMSNorm.
        dropout_prob (float): Dropout probability for regularization.
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
        use_qkv_bias: bool,
        use_o_bias: bool,
        use_qkv_proj: bool,
        max_temporal_len: int,  # add to RoPE or remove.
        max_spatial_len: int,   # add to RoPE or remove
        rms_norm_eps: float,
        dropout_prob: float,
        device: torch.device,
        dtype: torch.dtype
    ):
        super().__init__()
        
        self.st_attention = SpatioTemporalAttention(
            num_heads=num_heads,
            d_model=d_model,
            query_groups=query_groups,
            rope_theta=rope_theta,
            softmax_scale=softmax_scale,
            use_qkv_bias=use_qkv_bias,
            use_o_bias=use_o_bias,
            use_qkv_proj=use_qkv_proj,
            max_temporal_len=max_temporal_len,
            max_spatial_len=max_spatial_len,
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
        use_qk_norm: bool = True,
        use_mqa: bool = False,
        qk_norm_eps: float = 1e-10,
        qk_norm_type: int = 2,
        is_causal: bool = 2,
        use_cache: bool = False,
        kv_cache: Optional[TemporalKVCache] = None,
        layer_idx: Optional[int] = None,
        padding_mask: Optional[Tensor] = None,
        attention_interleave: Literal["st", "ts"] = "st"
    ) -> Tensor:
        """Forward pass of ST transformer block.
        
        Args:
            x (Tensor): Input tensor of shape [B, T_frames, H*W, d_model].
            use_qk_norm (bool): Whether to use QK normalization or not.
            use_mqa (bool): Whether to use MQA or not.
            qk_norm_eps (float): Epsilon value for QK norm.
            qk_norm_type (int): Type of normalization for QK norm.
            is_causal (bool): Whether to use causal masking or not.
            use_cache (bool): Whether to use cache or not.
            kv_cache (Optional[TemporalKVCache]): KV caching module.
            layer_idx (Optional[int]): Layer index to update KVs.
            padding_mask (Optional[Tensor]): Padding tensor of shape [B, T_frames.]
            attention_interleave (Literal["st", "ts"]): Interleaving method.
                st: Sequential application as TA(SA(x)).
                ts: Sequential application as SA(TA(x)).

        Returns:
            Tensor: Output tensor of shape [B, T_frames, H*W, d_model].
        """
        with autocast(device_type=x.device.type):
            return x + self.dropout(
                self.st_attention(
                    self.rms_norm(x),
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

def test_attention(use_pad:bool=True, return_debug:bool=False):
    num_heads, d_model, query_groups, rope_theta = 4, 64, 2, 10000.0
    softmax_scale = 1/(d_model//num_heads)**0.5
    attn = SpatioTemporalAttention(
        num_heads, d_model, query_groups, rope_theta,
        softmax_scale, False, False, True, 1000, 1000,
        device="cpu", dtype=torch.float32
    )
    B, T_frames, H, W = 4, 6, 16, 16
    x = torch.randn(B, T_frames, H*W, d_model, device="cpu", dtype=torch.float32)
    padding_mask = None
    if use_pad:
        padding_mask = torch.randint(
            0, 2, (B, T_frames), device="cpu", dtype=torch.bool
        )
    if return_debug:
        out, query_s, key_s, value_s, query_t, key_t, value_t = attn(
            x, padding_mask=padding_mask, attention_interleave="st", _return_debug=True
        )
        return out, query_s, key_s, value_s, query_t, key_t, value_t
    out = attn(x, padding_mask=padding_mask, attention_interleave="st")
    loss = out.sum()
    loss.backward()
    for name, param in attn.named_parameters():
        print(f"{name}: {param.grad}")
    return out

def test_attention_block(use_pad:bool=True):
    num_heads, d_model, query_groups, rope_theta = 4, 64, 2, 10000.0
    softmax_scale = 1/(d_model//num_heads)**0.5
    dropout_prob, rms_norm_eps = 0.2, 1e-10
    attn = SpatioTemporalAttentionBlock(
        num_heads, d_model, query_groups, rope_theta,
        softmax_scale, False, False, True, 1000, 1000,
        rms_norm_eps=rms_norm_eps, dropout_prob=dropout_prob,
        device="cpu", dtype=torch.float32
    )
    B, T_frames, H, W = 4, 6, 16, 16
    x = torch.randn(B, T_frames, H*W, d_model, device="cpu", dtype=torch.float32)
    padding_mask = None
    if use_pad:
        padding_mask = torch.randint(
            0, 2, (B, T_frames), device="cpu", dtype=torch.bool
        )
    out = attn(x, padding_mask=padding_mask, attention_interleave="st")
    loss = out.sum()
    loss.backward()
    for name, param in attn.named_parameters():
        print(f"{name}: {param.grad}")
    return out
