import pytest
import torch

from configs.text_transformer.model_args.xsmall import ModelArgs
from models.text_transformer.kv_cache import KVCache
from models.text_transformer.attention import CausalAttention
from models.text_transformer.attention import CausalAttentionBlock
from models.text_transformer.block import CausalTransformerBlock
from models.text_transformer.text_transformer import CausalTransformer

B, T_tokens, D = 10, 15, ModelArgs().d_model
num_heads = ModelArgs().num_heads
head_dim = D//num_heads
layer_idx = 1

@pytest.fixture
def model_args():
    return ModelArgs()

@pytest.fixture
def kv_cache(model_args):
    return KVCache(
        num_heads=model_args.num_heads,
        num_layers=model_args.num_layers,
        head_dim=head_dim,
        max_batch_size=model_args.max_batch_size,
        max_seq_len=model_args.max_seq_len,
        device=model_args.device,
        dtype=model_args.dtype
    )

@pytest.fixture
def attn(model_args):
    return CausalAttention(
        d_model=D,
        num_heads=model_args.num_heads,
        query_groups=model_args.query_groups,
        use_qkv_bias=model_args.use_qkv_bias,
        use_o_bias=model_args.use_o_bias,
        use_qkv_proj=model_args.use_qkv_proj,
        rope_theta=model_args.rope_theta,
        softmax_scale=model_args.softmax_scale,
        device=model_args.device,
        dtype=model_args.dtype
    )

@pytest.fixture
def attn_block(model_args):
    return CausalAttentionBlock(
        d_model=D,
        num_heads=model_args.num_heads,
        query_groups=model_args.query_groups,
        use_qkv_bias=model_args.use_qkv_bias,
        use_o_bias=model_args.use_o_bias,
        use_qkv_proj=model_args.use_qkv_proj,
        rope_theta=model_args.rope_theta,
        softmax_scale=model_args.softmax_scale,
        rms_norm_eps=model_args.rms_norm_eps,
        dropout_prob=model_args.dropout_prob,
        device=model_args.device,
        dtype=model_args.dtype
    )

@pytest.fixture
def block(model_args):
    return CausalTransformerBlock(
        d_model=D,
        num_heads=model_args.num_heads,
        query_groups=model_args.query_groups,
        use_qkv_bias=model_args.use_qkv_bias,
        use_o_bias=model_args.use_o_bias,
        use_qkv_proj=model_args.use_qkv_proj,
        rope_theta=model_args.rope_theta,
        softmax_scale=model_args.softmax_scale,
        rms_norm_eps=model_args.rms_norm_eps,
        dropout_prob=model_args.dropout_prob,
        d_ffn=model_args.d_ffn,
        use_mlp_bias=model_args.use_mlp_bias,
        device=model_args.device,
        dtype=model_args.dtype
    )

@pytest.fixture
def model(model_args):
    return CausalTransformer(model_args)

@pytest.fixture
def x(model_args):
    torch.manual_seed(0)
    return torch.randn(
        B, T_tokens, D, device=model_args.device, dtype=model_args.dtype
    )

@pytest.fixture
def input_ids(model_args):
    torch.manual_seed(0)
    return torch.randint(
        0, model_args.vocab_size, (B, T_tokens),
        device=model_args.device, dtype=torch.int64
    )

@pytest.fixture
def padding_mask(model_args):
    torch.manual_seed(0)
    return torch.randint(
        0, 2, (B, T_tokens), device=model_args.device, dtype=torch.bool
    )

def test_attn_no_cache(attn, x, padding_mask):
    out1 = attn(x)
    out2 = attn(x, padding_mask=padding_mask)
    for out in [out1, out2]:
        assert out.shape == (B, T_tokens, D)

def test_attn_block_no_cache(attn_block, x, padding_mask):
    out1 = attn_block(x)
    out2 = attn_block(x, padding_mask=padding_mask)
    for out in [out1, out2]:
        assert out.shape == (B, T_tokens, D)

def test_block_no_cache(block, x, padding_mask):
    out1 = block(x)
    out2 = block(x, padding_mask=padding_mask)
    for out in [out1, out2]:
        assert out.shape == (B, T_tokens, D)

def test_model_no_cache(model, input_ids, padding_mask):
    out1 = model(input_ids)
    out2 = model(input_ids, padding_mask=padding_mask)
    for out in [out1, out2]:
        assert out.shape == (B, T_tokens, ModelArgs().vocab_size)

def test_attn_cache(attn, kv_cache):
    for t in range(T_tokens):
        x_t = torch.randn(
            B, 1, D, device=ModelArgs().device, dtype=torch.float32
        )
        padding_mask_t = torch.randint(
            0, 2, (B, 1), device=ModelArgs().device, dtype=torch.bool
        )
        out = attn(
            x_t,
            use_cache=True,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            is_causal=True,
            padding_mask=padding_mask_t
        )
        past_k, past_v = kv_cache.get(layer_idx)
        if past_k is not None and past_v is not None:
            assert past_k.shape == (B, num_heads, t+1, head_dim)
            assert past_v.shape == (B, num_heads, t+1, head_dim)

def test_attn_block_cache(attn_block, kv_cache):
    for t in range(T_tokens):
        x_t = torch.randn(
            B, 1, D, device=ModelArgs().device, dtype=torch.float32
        )
        padding_mask_t = torch.randint(
            0, 2, (B, 1), device=ModelArgs().device, dtype=torch.bool
        )
        out = attn_block(
            x_t,
            use_cache=True,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            is_causal=True,
            padding_mask=padding_mask_t
        )
        past_k, past_v = kv_cache.get(layer_idx)
        if past_k is not None and past_v is not None:
            assert past_k.shape == (B, num_heads, t+1, head_dim)
            assert past_v.shape == (B, num_heads, t+1, head_dim)

def test_block_cache(block, kv_cache):
    for t in range(T_tokens):
        x_t = torch.randn(
            B, 1, D, device=ModelArgs().device, dtype=torch.float32
        )
        padding_mask_t = torch.randint(
            0, 2, (B, 1), device=ModelArgs().device, dtype=torch.bool
        )
        out = block(
            x_t,
            use_cache=True,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            is_causal=True,
            padding_mask=padding_mask_t
        )
        past_k, past_v = kv_cache.get(layer_idx)
        if past_k is not None and past_v is not None:
            assert past_k.shape == (B, num_heads, t+1, head_dim)
            assert past_v.shape == (B, num_heads, t+1, head_dim)

def test_model_cache(model, kv_cache):
    for t in range(T_tokens):
        input_ids_t = torch.randint(
            0, ModelArgs().vocab_size, (B, 1), device=ModelArgs().device, dtype=torch.int64
        )
        padding_mask_t = torch.randint(
            0, 2, (B, 1), device=ModelArgs().device, dtype=torch.bool
        )
        out = model(input_ids_t, padding_mask=padding_mask_t, use_cache=True)
        past_k, past_v = kv_cache.get(layer_idx)
        if past_k is not None and past_v is not None:
            assert past_k.shape == (B, num_heads, t+1, head_dim)
            assert past_v.shape == (B, num_heads, t+1, head_dim)
