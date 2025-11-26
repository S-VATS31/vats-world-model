from typing import Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast

from utils.attention_utils import apply_qk_norm, extend_kv_heads
from models.s_transformer.spatial_kv_cache import SpatialKVCache

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class RoPE2D(nn.Module):
    """2D Rotary Positional Embeddings using additive frequencies.

    Args:
        head_dim (int): Dimensionality of each attention head.
        rope_theta (float): Exponential base for inverse frequency.
        device (torch.device): Accelerator at use.
        dtype (torch.dtype): Data type of model parameters.
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

        # inv_freq: [head_dim//2]
        inv_freq = 1.0 / (
            rope_theta ** (
                torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """Apply 2D RoPE using combined rotation.

        Args:
            x (Tensor): Input query or key tensor of shape [B, L, num_heads, head_dim].
            H (int): Height of spatial grid (H_p).
            W (int): Width of spatial grid (W_p).

        Returns:
            Tensor: Output tensor of same shape with RoPE applied.
        """
        B, num_spatial_patches, num_heads, _ = x.shape
        assert x.size(-1) == self.head_dim

        pos_h = torch.arange(H, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        pos_w = torch.arange(W, device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        freqs_h = torch.einsum("i,j->ij", pos_h, self.inv_freq)
        freqs_w = torch.einsum("i,j->ij", pos_w, self.inv_freq)
        
        freqs_h_grid = freqs_h[:, None, :].expand(-1, W, -1)
        freqs_w_grid = freqs_w[None, :, :].expand(H, -1, -1)

        freqs_combined = freqs_h_grid + freqs_w_grid

        cos_theta = freqs_combined.cos() # [H, W, head_dim//2]
        sin_theta = freqs_combined.sin() # [H, W, head_dim//2]
        
        cos_theta = cos_theta[None, ..., None, :]
        sin_theta = sin_theta[None, ..., None, :]

        x_spatial = x.view(B, H, W, num_heads, self.head_dim)
        x1, x2 = x_spatial[..., ::2], x_spatial[..., 1::2]

        # Apply the single combined rotation
        x_rotated = torch.cat(
            [x1 * cos_theta - x2 * sin_theta, 
             x1 * sin_theta + x2 * cos_theta], dim=-1
        )
        
        return x_rotated.view(B, num_spatial_patches, num_heads, self.head_dim)


class SpatialAttention(nn.Module):
    """Spatial attention module.
    
    Args:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads.
        query_groups (int): Number of query groups for GQA.
        rope_theta (float): Exponential base for RoPE.
        softmax_scale (float): Softmax scale for attention computation.
        use_qkv_bias (bool): Whether to use QKV bias or not.
        use_o_bias (bool): Whether to use output projection bias or not.
        use_qkv_proj (bool): Whether to use QKV projection or not.
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
        device: torch.device,
        dtype: torch.dtype
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model must be divsible by num_heads, got {d_model} % {num_heads} != 0"
            )
        if num_heads % query_groups != 0:
            raise ValueError(
                f"num_heads must be divisble by query_groups, got {num_heads} % {query_groups}"
            )
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.query_groups = query_groups
        self.softmax_scale = softmax_scale
        self.use_qkv_proj = use_qkv_proj
        self.device = device
        self.dtype = dtype
        self.head_dim = d_model // num_heads
        self.heads_per_group = num_heads // query_groups

        # QKV projections
        if use_qkv_proj:
            self.w_qkv = nn.Linear(
                d_model,
                num_heads*self.head_dim + 2*query_groups*self.head_dim,
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

        # Output projection
        self.w_o = nn.Linear(
            d_model,
            d_model,
            bias=use_o_bias,
            device=device,
            dtype=dtype
        )

        # RoPE setup
        self.rope = RoPE2D(
            head_dim=self.head_dim,
            rope_theta=rope_theta,
            device=device,
            dtype=dtype
        )

    def _setup_qkv(
        self,
        x: Tensor,
        H_patch: int,
        W_patch: int,
        use_qk_norm: bool = True,
        use_mqa: bool = False,
        qk_norm_eps: float = 1e-10,
        qk_norm_type: int = 2,
        use_cache: bool = False,
        layer_idx: Optional[int] = None,
        kv_cache: Optional[SpatialKVCache] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Setup spatial QKV tensors.
        
        Args:
            x (Tensor): Input tensor of shape [B, H*W, d_model].
            H_patch (int): Height of each patch.
            W_patch (int): Width of each patch.
            use_qk_norm (bool): Whether to use QK normalization or not.
            use_mqa (bool): Whether to use MQA or not.
            qk_norm_eps (float): Epsilon value for QK normalization.
            qk_norm_type (int): Type of normalization.
            use_cache (bool): Whether to use KV caching or not.
            layer_idx (Optional[int]): Layer index for KV caching.
            kv_cache (Optional[SpatialKVCache]): KV caching module.

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                Tensor: Query tensor of shape [B, num_heads, H*W, head_dim].
                Tensor: Key tensor of shape [B, num_heads or 1, H*W, head_dim].
                Tensor: Value tensor of shape [B, num_heads or 1, H*W, head_dim].
        """
        B, num_spatial_patches, _ = x.shape

        # Get QKV tensors
        if self.use_qkv_proj:
            qkv = self.w_qkv(x)
            q, kv = torch.split(
                qkv,
                [self.num_heads*self.head_dim, 2*self.query_groups*self.head_dim],
                dim=-1
            )
            k, v = kv.chunk(2, dim=-1)
        else:
            q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        # Reshape to 4D
        q = q.view(B, num_spatial_patches, self.num_heads, self.head_dim)
        k = k.view(B, num_spatial_patches, self.query_groups, self.head_dim)
        v = v.view(B, num_spatial_patches, self.query_groups, self.head_dim)

        # Apply QK normalization
        if use_qk_norm:
            q = apply_qk_norm(q, eps=qk_norm_eps, norm=qk_norm_type)
            k = apply_qk_norm(k, eps=qk_norm_eps, norm=qk_norm_type)

        # Apply 2D RoPE
        q = self.rope(q, H_patch, W_patch)
        k = self.rope(k, H_patch, W_patch)

        # Extend KV heads
        k = extend_kv_heads(k, dim=2, repeats=self.heads_per_group, use_mqa=use_mqa)
        v = extend_kv_heads(v, dim=2, repeats=self.heads_per_group, use_mqa=use_mqa)

        # Handle KV cache
        if use_cache and kv_cache is not None and layer_idx is not None:
            # k_new:, v_new: [B, num_heads, H*W, head_dim]
            k_new = k.permute(0, 2, 1, 3)
            v_new = v.permute(0, 2, 1, 3)
            past_k, past_v = kv_cache.get_cached_kv(layer_idx, patches=None)
            kv_cache.update(layer_idx, k_new, v_new)
            if past_k is not None and past_v is not None:
                k_new = torch.cat([k_new, past_k], dim=2)
                v_new = torch.cat([v_new, past_v], dim=2)

            return q.permute(0, 2, 1, 3), k_new, v_new

        # q: [B, num_heads, H*W, d_model]
        # k, v: [B, num_heads or 1, H*W, d_model]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        return q, k, v

    def _spatial_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        is_causal: bool = True,
        padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Apply spatial attention.
        
        Args:
            query (Tensor): Query tensor of shape [B, num_heads, T_q, d_model]. T_q=H*W.
            key (Tensor): Key tensor of shape [B, num_heads or 1, T_k, d_model]. T_k=H*W.
            value (Tensor): Value tensor of shape [B, num_heads or 1, H*W, d_model].
            is_causal (bool): Whether to use causal masking or not.
            padding_mask (Optional[Tensor]): Padding tensor of shape [B, T_q].
        
        Returns:
            Tensor: Output tensor of shape [B, H*W, d_model].
        """
        B, _, T_q, _ = query.shape
        T_k = key.size(2)

        # Handle padding mask
        if padding_mask is not None:
            padding_mask = padding_mask.bool() # [B, T_q]
            query_padding_mask = padding_mask[:, None, :, None] # [B, 1, T_q, 1]
            # Handle case where T_k>T_q (cache being used)
            if T_k > T_q:
                past_padding_mask = torch.ones(
                    B, T_k - T_q, 
                    device=self.device, 
                    dtype=torch.bool
                )
                full_key_mask = torch.cat([past_padding_mask, padding_mask], dim=1) # [B, T_k]
                key_padding_mask = full_key_mask[:, None, None, :] # [B, 1, 1, T_k]
            else:
                key_padding_mask = padding_mask[:, None, None, :] # [B, 1, 1, T_k]
            attn_mask = torch.logical_or(query_padding_mask, key_padding_mask)
            if is_causal:
                # causal_mask: [H*W, H*W], True: pad, False: compute attention
                causal_mask = torch.triu(
                    torch.ones(T_q, T_k, device=self.device, dtype=torch.bool),
                    diagonal=1
                )
                causal_mask = causal_mask[None, None, :, :] # [1, 1, T_q, T_k]
                attn_mask = torch.logical_or(attn_mask, causal_mask)
        else:
            if is_causal:
                # Causal mask is only constraint
                causal_mask = torch.triu(
                    torch.ones(T_q, T_k, device=self.device, dtype=torch.bool),
                    diagonal=1
                )
                attn_mask = causal_mask[None, None, :, :] # [1, 1, T_q, T_k]
            attn_mask = None

        # Compute attention
        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask
        ) # [B, num_heads, H*W, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        
        return self.w_o(out)

    def forward(
        self, 
        x: Tensor,
        H_patch: int,
        W_patch: int,
        use_qk_norm: bool = True,
        use_mqa: bool = False,
        qk_norm_eps: float = 1e-10,
        qk_norm_type: int = 2,
        use_cache: bool = False,
        layer_idx: Optional[int] = None,
        kv_cache: Optional[SpatialKVCache] = None,
        is_causal: bool = True,
        padding_mask: Optional[Tensor] = None,
        *,
        _return_qkv: bool = False
    ) -> Union[Tuple[Tensor, ...], Tensor]:
        """Forward pass of spatial attention layer.

        Args:
            x (Tensor): Input tensor of shape [B, H*W, d_model].
            H_patch (int): Height of image patch.
            W_patch (int): Width of image patch.
            use_qk_norm (bool): Whether to use QK normalization or not.
            use_mqa (bool): Whether to use MQA or not.
            qk_norm_eps (float): Epsilon value for QK normalization.
            use_cache (bool): Whether to use spatial KV caching or not.
            layer_idx (Optional[int]): Layer index for KV cache.
            kv_cache (Optional[SpatialKVCache]): KV caching module.
            is_causal (bool): Whether to apply causal masking or not.
            padding_mask (Optional[Tensor]): Padding tensor of shape [B, H*W].
            _return_qkv (bool): Whether to return QKV tensors.

        Returns:
            Tensor: Output tensor of shape [B, H*W, d_model].
        """
        with autocast(device_type=x.device.type):
            query, key, value = self._setup_qkv(
                x,
                H_patch=H_patch,
                W_patch=W_patch,
                use_qk_norm=use_qk_norm,
                use_mqa=use_mqa,
                qk_norm_eps=qk_norm_eps,
                qk_norm_type=qk_norm_type,
                use_cache=use_cache,
                layer_idx=layer_idx,
                kv_cache=kv_cache
            )
            out = self._spatial_attention(
                query=query, 
                key=key, 
                value=value, 
                is_causal=is_causal, 
                padding_mask=padding_mask
            )
            if _return_qkv:
                return out, query, key, value
            return out


class SpatialAttentionBlock(nn.Module):
    """Spatial attention block.

    Args:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads.
        query_groups (int): Number of query groups for GQA.
        rope_theta (float): Exponential base for RoPE.
        softmax_scale (float): Softmax scale for attention computation.
        use_qkv_bias (bool): Whether to use QKV bias or not.
        use_o_bias (bool): Whether to use output projection bias or not.
        use_qkv_proj (bool): Whether to use QKV projection or not.
        rms_norm_eps (float): Epsilon value for RMSNorm.
        dropout_prob (float): Dropout probability for residual dropout.
        device (torch.device): Accelerator device used for module parameters.
        dtype (torch.dtype): Data type used for module parameters.
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
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        
        self.attn = SpatialAttention(
            d_model=d_model,
            num_heads=num_heads,
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
        use_qk_norm: bool = True,
        use_mqa: bool = False,
        qk_norm_eps: float = 1e-10,
        qk_norm_type: int = 2,
        use_cache: bool = False,
        layer_idx: Optional[int] = None,
        kv_cache: Optional[SpatialKVCache] = None,
        is_causal: bool = True,
        padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass of the spatial attention block.

        Args:
            x (Tensor): Input tensor of shape [B, H*W, d_model].
            H_patch (int): Height of each patch.
            W_patch (int): Width of each image patch.
            use_qk_norm (bool): Whether to use QK normalization or not.
            use_mqa (bool): Whether to use MQA or not.
            qk_norm_eps (float): Epsilon value for QK normalization.
            use_cache (bool): Whether to use spatial KV caching or not.
            layer_idx (Optional[int]): Layer index for KV cache.
            kv_cache (Optional[SpatialKVCache]): KV caching module.
            is_causal (bool): Whether to apply causal masking or not.
            padding_mask (Optional[Tensor]): Padding tensor of shape [B, H*W].

        Returns:
            Tensor: Output tensor of shape [B, H*W, d_model].
        """
        with autocast(device_type=x.device.type):
            return x + self.dropout(
                self.attn(
                    self.rms_norm(x),
                    H_patch=H_patch,
                    W_patch=W_patch,
                    use_qk_norm=use_qk_norm,
                    use_mqa=use_mqa,
                    qk_norm_eps=qk_norm_eps,
                    qk_norm_type=qk_norm_type,
                    use_cache=use_cache,
                    layer_idx=layer_idx,
                    kv_cache=kv_cache,
                    is_causal=is_causal,
                    padding_mask=padding_mask,
                )
            )


def test_attention(use_pad:bool=True, return_qkv:bool=False):
    d_model, num_heads, query_groups, rope_theta = 128, 32, 8, 10000.0
    softmax_scale = 1/(d_model//num_heads)**0.5
    attn = SpatialAttention(
        d_model, num_heads, query_groups, rope_theta,
        softmax_scale, False, False, True, 
        device="cpu", dtype=torch.float32
    )
    B, H, W = 16, 36, 72
    x = torch.randn(B, H*W, d_model, device="cpu", dtype=torch.float32)
    if use_pad:
        padding_mask = torch.randint(
            0, 2, (B, H*W), device="cpu", dtype=torch.bool
        )
    else:
        padding_mask = None
    if return_qkv:
        out, q, k, v = attn(x, H_patch=H, W_patch=W, padding_mask=padding_mask, _return_qkv=True)
        return out, q, k, v
    out =  attn(x, H_patch=H, W_patch=W, padding_mask=padding_mask)
    loss = out.sum()
    loss.backward()
    for name, param in attn.named_parameters():
        print(f"{name}: {param.grad}")
    return out

def test_attention_block(use_pad:bool=True):
    d_model, num_heads, query_groups, rope_theta = 128, 32, 8, 10000.0
    softmax_scale = 1/(d_model//num_heads)**0.5
    dropout, rms_norm_eps = 0.2, 1e-10
    attn = SpatialAttentionBlock(
        d_model, num_heads, query_groups, rope_theta,
        softmax_scale, False, False, True,
        rms_norm_eps=rms_norm_eps, dropout_prob=dropout,
        device="cpu", dtype=torch.float32
    )
    B, H, W = 16, 36, 72
    x = torch.randn(B, H*W, d_model, device="cpu", dtype=torch.float32)
    if use_pad:
        padding_mask = torch.randint(
            0, 2, (B, H*W), device="cpu", dtype=torch.bool
        )
    else:
        padding_mask = None
    out =  attn(x, H_patch=H, W_patch=W, padding_mask=padding_mask)
    loss = out.sum()
    loss.backward()
    for name, param in attn.named_parameters():
        print(f"{name}: {param.grad}")
    return out

def test_cache():
    d_model, num_heads, query_groups, rope_theta = 128, 32, 8, 10000.0
    softmax_scale = 1/(d_model//num_heads)**0.5
    attn = SpatialAttention(
        d_model, num_heads, query_groups, rope_theta,
        softmax_scale, False, False, True, 
        device="cpu", dtype=torch.float32
    )
    batch_size = 20
    num_layers, layer_idx = 5, 1
    H, W = 6, 6
    total_patches = H * W
    cache = SpatialKVCache(
        num_heads=num_heads,
        head_dim=d_model//num_heads,
        max_batch_size=50,
        max_patches=40,
        num_layers=num_layers,
        device="cpu",
        dtype=torch.float32
    )
    for t in range(1, 11):
        x = torch.randn(batch_size, total_patches, d_model)
        out = attn(
            x,
            H_patch=H,
            W_patch=W,
            use_cache=True,
            layer_idx=layer_idx,
            kv_cache=cache,
        )
        past_k, past_v = cache.get_cached_kv(layer_idx)
        if past_k is not None:
            assert past_k.shape == (
                batch_size, num_heads, cache.current_patches, d_model//num_heads
            )
            assert past_v.shape == (
                batch_size, num_heads, cache.current_patches, d_model//num_heads
            )
        assert out.shape[0] == batch_size
        assert out.shape[2] == d_model

if __name__ == "__main__":
    test_cache()
