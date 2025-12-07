from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast

from models.rope1d import RoPE
from models.text_transformer.kv_cache import KVCache
from utils.attention_utils import extend_kv_heads, apply_qk_norm

class CausalAttention(nn.Module):
    """Causal attention module.
    
    Args:
        d_model (int): Dimensionality of model embeddings.
        num_heads (int): Number of attention heads.
        query_groups (int): Number of query groups for GQA.
        use_qkv_bias (bool): Whether to use QKV bias or not.
        use_o_bias (bool): Whether to use output bias or not.
        use_qkv_proj (bool): Whether to use full QKV projection or not.
        rope_theta (float): Exponential base for RoPE.
        softmax_scale (float): Scaler for attention computation.
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
                f"num_heads must be divisble by query_groups, got {num_heads} % {query_groups} != 0"
            )
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.query_groups = query_groups
        self.use_qkv_proj = use_qkv_proj
        self.softmax_scale = softmax_scale
        self.device = device
        self.dtype = dtype
        self.head_dim = d_model // num_heads
        self.heads_per_group = num_heads // query_groups


        # Set up QKV projections
        if self.use_qkv_proj:
            self.w_qkv = nn.Linear(
                d_model,
                num_heads*self.head_dim+2*query_groups*self.head_dim,
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
        self.w_o = nn.Linear(
            d_model, d_model, bias=use_o_bias, device=device, dtype=dtype
        )

        # Set up RoPE
        self.rope = RoPE(self.head_dim, rope_theta, device, dtype)

    def _setup_qkv(
        self,
        x: Tensor,
        use_qk_norm: bool = True,
        use_mqa: bool = False,
        qk_norm_eps: float = 1e-10,
        qk_norm_type: int = 2,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None,
        use_cache: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Setup QKV tensors for causal attention.
        
        Args:
            x (Tensor): Input tensor of shape [B, T, d_model].
            use_qk_norm (bool): Whether to use QK normalization.
            use_mqa (bool): Whether to use MQA or not.
            qk_norm_eps (float): Epsilon value for normalization.
            qk_norm_type (int): Type of normalization.
            kv_cache (Optional[KVCache]): KV caching module.
            layer_idx (Optional[int]): Layer index to update KVs with respec to.
            use_cache (bool): Whether to use KV caching or not.

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - Tensor: Query tensor of shape [B, num_heads, T, head_dim].
                - Tensor: Key tensor of shape [B, num_heads or 1, T, head_dim].
                - Tensor: Value tensor of shape [B, num_heads or 1, T, head_dim].
        """
        B, T, _ = x.shape

        # Handle zero input tokens
        if T == 0:
            return (
                torch.empty(B, self.num_heads, T, self.head_dim, device=self.device, dtype=self.dtype),
                torch.empty(B, self.num_heads, T, self.head_dim, device=self.device, dtype=self.dtype),
                torch.empty(B, self.num_heads, T, self.head_dim, device=self.device, dtype=self.dtype)
            )
        
        # Get QKV tensors
        if self.use_qkv_proj:
            qkv = self.w_qkv(x)
            q, kv = torch.split(
                qkv, [self.num_heads*self.head_dim, 2*self.query_groups*self.head_dim], dim=-1
            )
            k, v = kv.chunk(2, dim=-1)
        else:
            q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        # Reshape to 4D tensors
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.query_groups, self.head_dim)
        v = v.view(B, T, self.query_groups, self.head_dim)

        # Apply QK normalization
        if use_qk_norm:
            q = apply_qk_norm(q, eps=qk_norm_eps, norm=qk_norm_type)
            k = apply_qk_norm(k, eps=qk_norm_eps, norm=qk_norm_type)

        # Apply RoPE
        q = self.rope(q)
        k = self.rope(k)

        # Extend KV heads
        k = extend_kv_heads(k, dim=2, repeats=self.heads_per_group, use_mqa=use_mqa)
        v = extend_kv_heads(v, dim=2, repeats=self.heads_per_group, use_mqa=use_mqa)

        # Handle KV caching if used
        if use_cache and kv_cache is not None and layer_idx is not None:
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            past_k, past_v = kv_cache.get(layer_idx, tokens=None) # None to get all current tokens
            kv_cache.update(layer_idx, k, v)
            if past_k is not None and past_v is not None:
                # Concatenate over sequence length dimension
                k = torch.cat([k, past_k], dim=2)
                v = torch.cat([v, past_v], dim=2)
            return q.permute(0, 2, 1, 3), k, v

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        return q, k, v

    def _scaled_dot_product_attention(
        self,
        x: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        is_causal: bool = True,
        padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Scaled dot product attention with masking.
        
        Args:
            x (Tensor): Input tensor of shape [B, T, d_model].
            query (Tensor): Query tensor of shape [B, num_heads, T, head_dim].
            key (Tensor): Key tensor of shape [B, num_heads or 1, T, head_dim].
            value (Tensor): Value tensor of shape [B, num_heads or 1, T, head_dim].
            is_causal (bool): Whether to use causal masking or not.
            padding_mask (Optional[Tensor]): Padding tensor of shape [B, T].

        Returns:
            Tensor: Attention output.
        """
        B, _, T_q, _ = query.shape
        T_k = key.size(2)

        # Handle empty tensors (T==0)
        if query.numel() == 0 or key.numel() == 0:
            return torch.empty_like(x, device=x.device, dtype=x.dtype)
        
        # Handle padding mask
        if padding_mask is not None:
            padding_mask = padding_mask.bool()
            query_padding_mask = padding_mask[:, None, :, None] # [B, 1, T_q, 1]
            # Handle T_k>T_q due to cache accumulation
            if T_k > T_q:
                past_padding_mask = torch.ones(
                    B, T_k-T_q,
                    device=padding_mask.device,
                    dtype=torch.bool
                )
                key_padding_mask_total = torch.cat([padding_mask, past_padding_mask], dim=-1) # [B, T_k]
                key_padding_mask = key_padding_mask_total[:, None, None, :] # [B, 1, 1, T_k]
            else:
                key_padding_mask = padding_mask[:, None, None, :] # [B, 1, 1, T_k]
            # attn_mask: [B, 1, T_q, T_k]
            attn_mask = torch.logical_or(query_padding_mask, key_padding_mask) # accumulate mask
            if is_causal:
                # causal_mask: [T_q, T_k]
                causal_mask = torch.triu(
                    torch.ones(T_q, T_k, device=self.device, dtype=torch.bool),
                    diagonal=1
                )
                causal_mask = causal_mask[None, None, :, :] # [1, 1, T_q, T_k]
                attn_mask = torch.logical_or(causal_mask, attn_mask) # [B, 1, T_q, T_k]
        else:
            if is_causal:
                # No padding, just causal masking
                causal_mask = torch.triu(
                    torch.ones(T_q, T_k, device=self.device, dtype=torch.bool),
                    diagonal=1
                )
                attn_mask = causal_mask[None, None, :, :] # [1, 1, T_q, T_k]
            attn_mask = None

        # Compute attention
        out = F.scaled_dot_product_attention(
            query=query, 
            key=key, 
            value=value, 
            attn_mask=attn_mask, 
            scale=self.softmax_scale
        ) # [B, num_heads, T_q, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T_q, -1) # [B, T_q, d_model]

        return self.w_o(out)

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
        padding_mask: Optional[Tensor] = None,
        *,
        _return_qkv: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
        """Forward pass of causal attention layer.
        
        Args:
            x (Tensor): Input tensor of shape [B, T, d_model].
            use_qk_norm (bool): Whether to use QK normalization.
            use_mqa (bool): Whether to use MQA or not.
            qk_norm_eps (float): Epsilon value for QK normalization.
            qk_norm_type (int): Type of normalization.
            kv_cache (Optional[KVCache]): KV caching module.
            layer_idx (Optional[int]): Layer index to update KVs with respect to.
            use_cache (bool): Whether to use KV caching or not.
            is_causal (bool): Whether to use causal masking or not.
            padding_mask (Optional[Tensor]): Padding tensor of shape [B, T, d_model].
            _return_qkv (bool): Whether to return QKV tensors.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
                - Tensor: Output tensor.
                - Tuple[Tensor, Tensor, Tensor, Tensor]: Output and QKV tensors. 
        """
        with autocast(device_type=x.device.type):
            q, k, v = self._setup_qkv(
                x,
                use_qk_norm=use_qk_norm,
                use_mqa=use_mqa,
                qk_norm_eps=qk_norm_eps,
                qk_norm_type=qk_norm_type,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
                use_cache=use_cache
            )
            out = self._scaled_dot_product_attention(
                x, q, k, v, is_causal=is_causal, padding_mask=padding_mask
            )
            if _return_qkv: 
                return out, q, k, v
            return out


class CausalAttentionBlock(nn.Module):
    """Causal attention block module.

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
        device (torch.device): Accelerator at use.
        dtype (torch.dtype): Data type of model parameters.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        query_groups: int,
        use_qkv_bias: False,
        use_o_bias: False,
        use_qkv_proj: bool,
        rope_theta: float,
        softmax_scale: float,
        rms_norm_eps: float,
        dropout_prob: float,
        device: torch.device,
        dtype: torch.dtype
    ):
        super().__init__()
        
        self.attn = CausalAttention(
            d_model=d_model,
            num_heads=num_heads,
            query_groups=query_groups,
            use_qkv_bias=use_qkv_bias,
            use_o_bias=use_o_bias,
            use_qkv_proj=use_qkv_proj,
            rope_theta=rope_theta,
            softmax_scale=softmax_scale,
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
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None,
        use_cache: bool = False,
        is_causal: bool = False,
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
            return x + self.dropout(
                self.attn(
                    self.rms_norm(x),
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

def test_attention(
    use_pad:bool=True, 
    return_qkv:bool=False, 
    log_grads:bool=False
):
    attn = CausalAttention(
        d_model=512,
        num_heads=32,
        query_groups=8,
        use_qkv_bias=False,
        use_o_bias=False,
        use_qkv_proj=True,
        rope_theta=10000.0,
        softmax_scale=0.25,
        device="cpu",
        dtype=torch.float32
    )
    B, T = 20, 15
    x = torch.randn(B, T, 512, device="cpu", dtype=torch.float32)
    if use_pad:
        padding_mask = torch.randint(
            0, 2, (B, T), device="cpu", dtype=torch.float32
        )
    else:
        padding_mask = None
    if return_qkv:
        out, q, k, v = attn(x, padding_mask=padding_mask, _return_qkv=True)
        if log_grads:
            loss = out.sum()
            loss.backward()
            for name, param in attn.named_parameters():
                print(f"{name}: {param.grad}")
        return out, q, k, v
    else:
        out = attn(x, padding_mask=padding_mask)
        if log_grads:
            loss = out.sum()
            loss.backward()
            for name, param in attn.named_parameters():
                print(f"{name}: {param.grad}")
        return out

def test_cache():
    d_model, num_heads, query_groups, rope_theta = 128, 32, 8, 10000.0
    softmax_scale = 1/(d_model//num_heads)**0.5
    attn = CausalAttentionBlock(
        d_model=d_model,
        num_heads=num_heads,
        query_groups=query_groups,
        use_qkv_bias=False,
        use_o_bias=False,
        use_qkv_proj=True,
        rope_theta=rope_theta,
        softmax_scale=softmax_scale,
        rms_norm_eps=1e-10,
        dropout_prob=0.0,
        device="cpu",
        dtype=torch.float32
    )
    B = 2
    num_layers, layer_idx = 3, 1
    T_max = 5
    cache = KVCache(
        num_heads=num_heads,
        num_layers=num_layers,
        head_dim=d_model//num_heads,
        max_batch_size=B,
        max_seq_len=T_max,
        device="cpu",
        dtype=torch.float32
    )
    for t in range(T_max):
        x_step = torch.randn(B, 1, d_model, device="cpu", dtype=torch.float32)
        out = attn(
            x_step,
            use_cache=True,
            layer_idx=layer_idx,
            kv_cache=cache,
            is_causal=True
        )
        past_k, past_v = cache.get(layer_idx)
        if past_k is not None and past_v is not None:
            assert past_k.shape == (B, num_heads, t+1, d_model//num_heads)
            assert past_v.shape == (B, num_heads, t+1, d_model//num_heads)
        assert out.shape[0] == B
        assert out.shape[2] == d_model
        print(f"Step {t}: out shape={out.shape}, cached K shape={past_k.shape if past_k is not None else None}")

if __name__ == "__main__":
    test_cache()

