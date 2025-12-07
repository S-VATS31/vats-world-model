import pytest
import torch

from configs.st_transformer.model_args.xsmall import ModelArgs
from models.st_transformer.temporal_cache import TemporalKVCache
from models.st_transformer.patch_embed import PatchEmbed3D
from models.st_transformer.st_attention import SpatioTemporalAttention
from models.st_transformer.st_attention import SpatioTemporalAttentionBlock
from models.st_transformer.st_block import SpatioTemporalTransformerBlock
from models.st_transformer.st_transformer import SpatioTemporalTransformer

B, T_frames, H, W, D = 15, 20, 12, 32, ModelArgs().d_model
num_heads = ModelArgs().num_heads
head_dim = D//num_heads
V = ModelArgs().codebook_size

@pytest.fixture
def model_args():
    return ModelArgs()

@pytest.fixture
def kv_cache(model_args):
    return TemporalKVCache(
        max_batch_size=model_args.max_batch_size,
        max_frames=model_args.max_frames,
        num_heads=model_args.num_heads,
        head_dim=model_args.d_model//model_args.num_heads,
        num_layers=model_args.num_layers,
        device=model_args.device,
        dtype=model_args.dtype
    )

@pytest.fixture
def patch_embed(model_args):
    return PatchEmbed3D(
        C_in=model_args.C_in,
        d_model=model_args.d_model,
        patch_size=model_args.patch_size,
        use_proj_bias=model_args.use_conv3d_bias,
        device=model_args.device,
        dtype=model_args.dtype
    )

@pytest.fixture
def attn(model_args):
    return SpatioTemporalAttention(
        num_heads=model_args.num_heads,
        d_model=model_args.d_model,
        query_groups=model_args.query_groups,
        rope_theta=model_args.rope_theta,
        softmax_scale=model_args.softmax_scale,
        use_qkv_bias=model_args.use_qkv_bias,
        use_o_bias=model_args.use_o_bias,
        use_qkv_proj=model_args.use_qkv_proj,
        device=model_args.device,
        dtype=model_args.dtype
    )

@pytest.fixture
def attn_block(model_args):
    return SpatioTemporalAttentionBlock(
        num_heads=model_args.num_heads,
        d_model=model_args.d_model,
        query_groups=model_args.query_groups,
        rope_theta=model_args.rope_theta,
        softmax_scale=model_args.softmax_scale,
        use_qkv_bias=model_args.use_qkv_bias,
        use_o_bias=model_args.use_o_bias,
        use_qkv_proj=model_args.use_qkv_proj,
        rms_norm_eps=model_args.rms_norm_eps,
        dropout_prob=model_args.dropout_prob,
        device=model_args.device,
        dtype=model_args.dtype
    )

@pytest.fixture
def block(model_args):
    return SpatioTemporalTransformerBlock(
        num_heads=model_args.num_heads,
        d_model=model_args.d_model,
        query_groups=model_args.query_groups,
        rope_theta=model_args.rope_theta,
        softmax_scale=model_args.softmax_scale,
        use_qkv_bias=model_args.use_qkv_bias,
        use_o_bias=model_args.use_o_bias,
        use_qkv_proj=model_args.use_qkv_proj,
        rms_norm_eps=model_args.rms_norm_eps,
        dropout_prob=model_args.dropout_prob,
        d_ffn=model_args.d_ffn,
        use_mlp_bias=model_args.use_mlp_bias,
        device=model_args.device,
        dtype=model_args.dtype
    )

@pytest.fixture
def model(model_args):
    return SpatioTemporalTransformer(model_args)

@pytest.fixture
def x(model_args):
    torch.manual_seed(0)
    return torch.randn(
        B, model_args.C_in, T_frames, H, W,
        device=model_args.device, dtype=model_args.dtype
    )

def test_attn_no_cache(attn, patch_embed, x):
    out1, THW_patches, mask = patch_embed(x)
    T_p, H_p, W_p = THW_patches
    out2 = attn(out1, H_p, W_p, T_p, padding_mask=mask)
    assert out2.shape == (B, T_p, H_p*W_p, D)

def test_attn_block_no_cache(attn_block, patch_embed, x):
    out1, THW_patches, mask = patch_embed(x)
    T_p, H_p, W_p = THW_patches
    out2 = attn_block(out1, H_p, W_p, T_p, padding_mask=mask)
    assert out2.shape == (B, T_p, H_p*W_p, D)

def test_block_no_cache(block, patch_embed, x):
    out1, THW_patches, mask = patch_embed(x)
    T_p, H_p, W_p = THW_patches
    out2 = block(out1, H_p, W_p, T_p, padding_mask=mask)
    assert out2.shape == (B, T_p, H_p*W_p, D)

def test_model_no_cache(model, patch_embed, x):
    _, THW_patches, _ = patch_embed(x)
    T_p, H_p, W_p = THW_patches
    logits = model(x)
    assert logits.shape == (B, T_p, H_p*W_p, V)

def test_attn_cache(attn, patch_embed, kv_cache):
    for t in range(T_frames):
        x_t = torch.randn(
            B, ModelArgs().C_in, 1, H, W,
            device=ModelArgs().device, dtype=ModelArgs().dtype
        )
        out_t, THW_patches, mask = patch_embed(x_t)
        _ = attn(
            out_t,
            THW_patches[1],
            THW_patches[2],
            THW_patches[0],
            use_cache=True,
            layer_idx=1,
            kv_cache=kv_cache,
            padding_mask=mask
        )
        past_k, past_v = kv_cache.get_cached_kv(1)
        if past_k is not None and past_v is not None:
            assert past_k.shape == (
                B*THW_patches[1]*THW_patches[2], num_heads, t+1, head_dim
            )
            assert past_v.shape == (
                B*THW_patches[1]*THW_patches[2], num_heads, t+1, head_dim
            )

def test_attn_block_cache(attn_block, patch_embed, kv_cache):
    for t in range(T_frames):
        x_t = torch.randn(
            B, ModelArgs().C_in, 1, H, W,
            device=ModelArgs().device, dtype=ModelArgs().dtype
        )
        out_t, THW_patches, mask = patch_embed(x_t)
        _ = attn_block(
            out_t,
            THW_patches[1],
            THW_patches[2],
            THW_patches[0],
            use_cache=True,
            layer_idx=1,
            kv_cache=kv_cache,
            padding_mask=mask
        )
        past_k, past_v = kv_cache.get_cached_kv(1)
        if past_k is not None and past_v is not None:
            assert past_k.shape == (
                B*THW_patches[1]*THW_patches[2], num_heads, t+1, head_dim
            )
            assert past_v.shape == (
                B*THW_patches[1]*THW_patches[2], num_heads, t+1, head_dim
            )

def test_block_cache(block, patch_embed, kv_cache):
    for t in range(T_frames):
        x_t = torch.randn(
            B, ModelArgs().C_in, 1, H, W,
            device=ModelArgs().device, dtype=ModelArgs().dtype
        )
        out_t, THW_patches, mask = patch_embed(x_t)
        _ = block(
            out_t,
            THW_patches[1],
            THW_patches[2],
            THW_patches[0],
            use_cache=True,
            layer_idx=1,
            kv_cache=kv_cache,
            padding_mask=mask
        )
        past_k, past_v = kv_cache.get_cached_kv(1)
        if past_k is not None and past_v is not None:
            assert past_k.shape == (
                B*THW_patches[1]*THW_patches[2], num_heads, t+1, head_dim
            )
            assert past_v.shape == (
                B*THW_patches[1]*THW_patches[2], num_heads, t+1, head_dim
            )

def test_model_cache(model, patch_embed, x, kv_cache):
    for t in range(T_frames):
        x_t = torch.randn(
            B, ModelArgs().C_in, 1, H, W,
            device=ModelArgs().device, dtype=ModelArgs().dtype
        )
        _, THW_patches, _ = patch_embed(x_t)
        logits = model(x, use_cache=True)
        past_k, past_v = kv_cache.get_cached_kv(1)
        if past_k is not None and past_v is not None:
            assert past_k.shape == (
                B*THW_patches[1]*THW_patches[2], num_heads, t+1, head_dim
            )
            assert past_v.shape == (
                B*THW_patches[1]*THW_patches[2], num_heads, t+1, head_dim
            )
            