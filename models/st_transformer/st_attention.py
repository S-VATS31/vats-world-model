from typing import Union, Tuple, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast

from utils.attention_utils import apply_qk_norm, extend_kv_heads
from models.st_transformer.temporal_cache import TemporalKVCache

class RoPE3D(nn.Module):
    def __init__(
        self,
        head_dim: int,
        rope_theta: float,
        device: torch.device,
        dtype: torch.dtype
    ):
        super().__init__()

        if head_dim % 3 != 0:
            raise ValueError(
                f"head_dim must be divisible by 3, got {head_dim} % 3 != 0"
            )
        
        self.head_dim = head_dim
        self.axis_dim = head_dim // 3 # Dimension per axis (T, H, W)
        inv_freq_thw = []
        for _ in range(3): # T, H, W
            inv_freq = 1.0 / (
                rope_theta ** (
                    torch.arange(0, self.axis_dim, 2, device=device, dtype=dtype) / self.axis_dim
                )
            )
            inv_freq_thw.append(inv_freq)
        # use False to prevent state_dict errors
        self.register_buffer("inv_freq_t", inv_freq_thw[0], persistent=False)
        self.register_buffer("inv_freq_h", inv_freq_thw[1], persistent=False)
        self.register_buffer("inv_freq_w", inv_freq_thw[2], persistent=False)

    def _apply_rope_spatial(
        self, 
        x: Tensor,
        H_patch: int, 
        W_patch: int
    ) -> Tensor:
        """Apply RoPE to spatial query and key tensors.
        
        Args:
            x (Tensor): Input spatial query or key tensor.
            H_patch (int): Height of each patch.
            W_patch (int): Width of each patch.

        Returns:
            Tensor: Rotated spatial query or key tensor.
        """
        batch_frames, num_spatial_patches, num_heads, d_model = x.shape
        x_spatial = x.view(batch_frames, H_patch, W_patch, num_heads, d_model)
        x_spatial = x_spatial.permute(0, 3, 1, 2, 4)  # [B*T, num_heads, H, W, d_model]

        pos_h = torch.arange(H_patch, device=x.device, dtype=x.dtype)
        pos_w = torch.arange(W_patch, device=x.device, dtype=x.dtype)

        freqs_h = torch.einsum("i,j->ij", pos_h, self.inv_freq_h)
        freqs_w = torch.einsum("i,j->ij", pos_w, self.inv_freq_w)

        freqs_h = freqs_h[:, None, :].expand(-1, W_patch, -1)
        freqs_w = freqs_w[None, :, :].expand(H_patch, -1, -1)
        freqs = freqs_h + freqs_w # sum over freqs

        cos = freqs.cos()[None, None, :, :, :]
        sin = freqs.sin()[None, None, :, :, :]

        x_axis = x_spatial[..., :self.axis_dim]
        x1 = x_axis[..., ::2]  # even pos (2i)
        x2 = x_axis[..., 1::2] # odd pos (2i+1)

        # rotation matrix
        rot = torch.cat([x1 * cos - x2 * sin,
                         x1 * sin + x2 * cos], dim=-1)

        x_out = torch.cat([rot, x_spatial[..., self.axis_dim:]], dim=-1)
        x_out = x_out.permute(0, 2, 3, 1, 4).contiguous().view(
            batch_frames, num_spatial_patches, num_heads, d_model
        )
        return x_out

    def _apply_rope_temporal(self, x: Tensor, T_patch: int) -> Tensor:
        """Apply RoPE to temporal query and key tensors.
        
        Args:
            x (Tensor): Input temporal query or key tensor.
            T_patch (int): Temporal length over each patch.
        
        Returns:
            Tensor: Rotated query or key tensor.
        """
        # x shape: [B*H*W, T_frames, num_heads, head_dim]
        pos_t = torch.arange(x.size(1), device=x.device, dtype=x.dtype)

        freqs_t = torch.einsum("i,j->ij", pos_t, self.inv_freq_t) # [T_patch, axis_dim//2]

        # Reshape cos/sin to match x dimensions: [1, T_patch, 1, axis_dim//2]
        cos = freqs_t.cos()[None, :, None, :]
        sin = freqs_t.sin()[None, :, None, :]

        x_t = x[..., :self.axis_dim]
        # x_t: [B*H*W, T_frames, num_heads, axis_dim]
        x1 = x_t[..., 0::2]
        x2 = x_t[..., 1::2]

        # rotation matrix
        rot = torch.cat([x1 * cos - x2 * sin,
                        x1 * sin + x2 * cos], dim=-1)

        # concatenate rotated + unrotated tensors
        x_out = torch.cat([rot, x[..., self.axis_dim:]], dim=-1)
        return x_out

    def forward(
        self,
        x: Tensor,
        H_patch: Optional[int] = None,
        W_patch: Optional[int] = None,
        T_patch: Optional[int] = None,
    ) -> Tensor:
        """Apply spatial or temporal RoPE.
        
        Args:
            x (Tensor): Input query or key tensor.
            H_patch (Optional[int]): Height of each patch.
            W_patch (Optional[int]): Width of each patch.
            T_patch (Optional[int]): Temporal length of each patch.

        Returns:
            Tensor: Output rotated spatial or temporal query or key tensor.
        """
        with autocast(device_type=x.device.type):
            if (
                T_patch is None and 
                H_patch is not None and 
                W_patch is not None
            ):
                return self._apply_rope_spatial(x, H_patch, W_patch)
            elif (
                H_patch is None and
                W_patch is None and
                T_patch is not None
            ):
                return self._apply_rope_temporal(x, T_patch)
            else:
                raise ValueError(
                    "Pass H_patch, W_patch for spatial RoPE, T_patch for temporal RoPE."
                )


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
        self.rope = RoPE3D(
            head_dim=self.head_dim,
            rope_theta=rope_theta,
            device=device,
            dtype=dtype
        )

    def _setup_spatial_qkv(
        self,
        x: Tensor,
        H_patch: int,
        W_patch: int,
        use_qk_norm: bool = True,
        use_mqa: bool = False,
        qk_norm_eps: float = 1e-10,
        qk_norm_type: int = 2
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Set up spatial QKV tensors.
        
        Args:
            x (Tensor): Input tensor of shape [B, T_frames, H*W, d_model].
            H_patch (int): Height of each patch.
            W_patch (int): Width of each patch.
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
        q = self.rope(q, H_patch, W_patch)
        k = self.rope(k, H_patch, W_patch)

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
        T_patch: int,
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
            T_patch (int): Temporal length of each patch.
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
        q = self.rope(q, T_patch=T_patch)
        k = self.rope(k, T_patch=T_patch)

        # Extend KV heads
        k = extend_kv_heads(k, dim=2, repeats=self.heads_per_group, use_mqa=use_mqa)
        v = extend_kv_heads(v, dim=2, repeats=self.heads_per_group, use_mqa=use_mqa)

        # Handle KV caching
        if use_cache and kv_cache is not None and layer_idx is not None:
            # k_new, v_new: [B*H*W, num_heads, T_frames, head_dim]
            k_new = k.permute(0, 2, 1, 3)
            v_new = v.permute(0, 2, 1, 3)
            past_k, past_v = kv_cache.get_cached_kv(layer_idx)
            kv_cache.update(layer_idx, k_new, v_new, num_spatial_patches)
            if past_k is not None and past_v is not None:
                k_new = torch.cat([past_k, k_new], dim=2)
                v_new = torch.cat([past_v, v_new], dim=2)
            
            # Use the concatenated KV for attention
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
        if query.numel() == 0 or key.numel() == 0:
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
        T_q = query.size(2)
        T_k = key.size(2)

        # Handle zero input frames
        if query.numel() == 0 or key.numel() == 0:
            return torch.empty_like(x, device=x.device, dtype=x.dtype)
        
        # Handle padding mask
        if padding_mask is not None:
            # [B, 1, T_frames]
            padding_mask = padding_mask.bool().unsqueeze(1).expand(B, num_spatial_patches, T_frames)
            padding_mask = padding_mask.contiguous().reshape(-1, T_frames)
            query_padding_mask = padding_mask[:, None, :, None] # [B, 1, T_frames, 1]
            if T_k > T_q:
                past_padding_mask = torch.ones(
                    B*num_spatial_patches, T_k - T_q,
                    device=self.device,
                    dtype=torch.bool
                )
                full_key_mask = torch.cat([past_padding_mask, padding_mask], dim=-1) # [B, T_k]
                key_padding_mask = full_key_mask[:, None, None, :] # [B, 1, 1, T_k]
            else:
                key_padding_mask = padding_mask[:, None, None, :] # [B, 1, 1, T_frames]
            attn_mask = torch.logical_or(query_padding_mask, key_padding_mask)
            
            # Handle temporal causal masking
            if is_causal:
                # causal_mask: [T_frames, T_frames], True = pad, False = compute attention
                causal_mask = torch.triu(
                    torch.ones(T_q, T_k, device=x.device, dtype=torch.bool),
                    diagonal=1
                )
                causal_mask = causal_mask[None, None, :, :] # [1, 1, T_frames, T_frames]
                attn_mask = torch.logical_or(attn_mask, causal_mask)
        else:
            # No padding, check for causal
            if is_causal:
                causal_mask = torch.triu(
                    torch.ones(T_q, T_k, device=x.device, dtype=torch.bool),
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
        attention_interleave: Literal["st", "ts"] = "st",
        *,
        _return_debug: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        """Forward pass of ST transformer.
        
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
            padding_mask (Optional[Tensor]): Padding tensor of shape [B, T_frames.]
            attention_interleave (Literal["st", "ts"]): Interleaving method.
                st: Sequential application as TA(SA(x)).
                ts: Sequential application as SA(TA(x)).
            _return_debug (bool): Whether to return QKV tensors for debugging.
                Returns attention output, spatial QKV, temporal QKV.

        Returns:
            Union:
                Tensor: Attention output.
                Tuple[Tensor, ...]: Attention output + QKV tensors.
        """
        with autocast(device_type=x.device.type):
            # Get temporal QKV
            query_temporal, key_temporal, value_temporal = self._setup_temporal_qkv(
                x,
                T_patch,
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
                H_patch,
                W_patch,
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
        H_patch: int,
        W_patch: int,
        T_patch: int,
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
        """Forward pass of ST attention block.
        
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
            return x + self.dropout(
                self.st_attention(
                    self.rms_norm(x),
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

def test_attention(use_pad:bool=True, return_debug:bool=False):
    num_heads, d_model, query_groups, rope_theta = 4, 96, 2, 10000.0
    softmax_scale = 1/(d_model//num_heads)**0.5
    attn = SpatioTemporalAttention(
        num_heads, d_model, query_groups, rope_theta,
        softmax_scale, False, False, True,
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
            x, H, W, T_frames, padding_mask=padding_mask, attention_interleave="st", _return_debug=True
        )
        return out, query_s, key_s, value_s, query_t, key_t, value_t
    out = attn(x, H, W, T_frames, padding_mask=padding_mask, attention_interleave="st")
    loss = out.sum()
    loss.backward()
    for name, param in attn.named_parameters():
        print(f"{name}: {param.grad}")
    return out

def test_attention_block(use_pad:bool=True):
    num_heads, d_model, query_groups, rope_theta = 4, 96, 2, 10000.0
    softmax_scale = 1/(d_model//num_heads)**0.5
    dropout_prob, rms_norm_eps = 0.2, 1e-10
    attn = SpatioTemporalAttentionBlock(
        num_heads, d_model, query_groups, rope_theta,
        softmax_scale, False, False, True,
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
    out = attn(
        x, H, W, T_frames, 
        use_cache=True, 
        padding_mask=padding_mask, 
        attention_interleave="st"
    )
    loss = out.sum()
    loss.backward()
    for name, param in attn.named_parameters():
        print(f"{name}: {param.grad}")
    return out

if __name__ == "__main__":
    out = test_attention_block(True)
    print(out.shape)
