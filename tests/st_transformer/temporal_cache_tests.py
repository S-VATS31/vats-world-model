import pytest
import torch

from configs.st_transformer.model_args.xsmall import ModelArgs
from models.st_transformer.patch_embed import PatchEmbed3D
from models.st_transformer.st_attention import SpatioTemporalAttention
from models.st_transformer.st_attention import SpatioTemporalAttentionBlock
from models.st_transformer.st_block import SpatioTemporalTransformerBlock
from models.st_transformer.st_transformer import SpatioTemporalTransformer

B, T_frames, H, W = 15, 20, 12, 32

@pytest.fixture
def model_args():
    return ModelArgs()

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
        query_groups=model_args.query_gorups,
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
        query_groups=model_args.query_gorups,
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
        query_groups=model_args.query_gorups,
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
    pass

def test_attn_block_no_cache(attn_block, patch_embed, x):
    pass

def test_block_no_cache(block, patch_embed, x):
    pass

def test_model_no_cache(model, patch_embed, x):
    pass

def test_attn_cache(attn, patch_embed, x):
    pass

def test_attn_block_cache(attn_block, patch_embed, x):
    pass

def test_block_cache(block, patch_embed, x):
    pass

def test_model_cache(model, patch_embed, x):
    pass