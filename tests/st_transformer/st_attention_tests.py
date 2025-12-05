import pytest
import torch
import random

from configs.st_transformer.model_args.xsmall import ModelArgs
from models.st_transformer.st_attention import SpatioTemporalAttention

B, T_frames, H, W, D = 20, 15, 12, 32, ModelArgs().d_model

@pytest.fixture
def model_args():
    return ModelArgs()

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
def x(model_args):
    torch.manual_seed(0)
    return torch.randn(
        B, T_frames, H*W, model_args.d_model,
        device=model_args.device, dtype=model_args.dtype
    )

@pytest.fixture
def padding_mask(model_args):
    torch.manual_seed(0)
    return torch.randint(
        0, 2, (B, T_frames), device=model_args.device, dtype=torch.bool
    )

def test_shape(attn, x, padding_mask):
    out = attn(x, H, W, T_frames, padding_mask=padding_mask)
    assert out.shape == (B, T_frames, H*W, D)

def test_no_padding(attn, x):
    out = attn(x, H, W, T_frames)
    assert out.shape == (B, T_frames, H*W, D)

def test_gradient_flow(attn, x, padding_mask):
    out1 = attn(x, H, W, T_frames, padding_mask=padding_mask)
    out2 = attn(x, H, W, T_frames)
    for out in [out1, out2]:
        loss = out.sum()
        loss.backward()
        for _, param in attn.named_parameters():
            assert torch.isfinite(param.grad).all()
            assert torch.isreal(param.grad).all()
            assert not torch.isnan(param.grad).any()

def test_numerical_stability(attn, x, padding_mask):
    out1 = attn(x, H, W, T_frames, padding_mask=padding_mask)
    out2 = attn(x, H, W, T_frames)
    for out in [out1, out2]:
        assert not out.isnan().any()
        assert not out.isinf().any()
        assert not out.norm().isnan().any()
        assert not out.norm().isinf().any()
        assert out.isfinite().all()
        assert out.isreal().all()
        assert out.norm().isfinite().all()
        assert out.norm().isreal().all()

def test_deterministic_output(attn, x, padding_mask):
    out1_mask = attn(x, H, W, T_frames, padding_mask=padding_mask)
    out2_mask = attn(x, H, W, T_frames, padding_mask=padding_mask)
    out1_no_mask = attn(x, H, W, T_frames)
    out2_no_mask = attn(x, H, W, T_frames)
    torch.testing.assert_close(out1_mask, out2_mask)
    torch.testing.assert_close(out1_no_mask, out2_no_mask)

def test_cache(attn, x, padding_mask):
    out1 = attn(x, H, W, T_frames, use_cache=True, padding_mask=padding_mask)
    out2 = attn(x, H, W, T_frames, use_cache=True)
    for out in [out1, out2]:
        assert out.shape == (B, T_frames, H*W, D)
        assert not out.isnan().any()
        assert not out.isinf().any()
        assert not out.norm().isnan().any()
        assert not out.norm().isinf().any()
        assert out.isfinite().all()
        assert out.isreal().all()
        assert out.norm().isfinite().all()
        assert out.norm().isreal().all()

def test_variable_batch_size(attn):
    for batch_size in [1, 2, 4, 8, 16, 32, 64]:
        x_new = torch.randn(
            batch_size, T_frames, H*W, D, 
            device=ModelArgs().device, dtype=ModelArgs().dtype
        )
        padding_mask_new = torch.randint(
            0, 2, (batch_size, T_frames), 
            device=ModelArgs().device, dtype=torch.bool
        )
        out = attn(x_new, H, W, T_frames, padding_mask=padding_mask_new)
        assert out.shape == (batch_size, T_frames, H*W, D)
        assert not out.isnan().any()
        assert not out.isinf().any()
        assert not out.norm().isnan().any()
        assert not out.norm().isinf().any()
        assert out.isfinite().all()
        assert out.isreal().all()
        assert out.norm().isfinite().all()
        assert out.norm().isreal().all()

def test_variable_frames(attn):
    for frames in range(1, 50):
        x_new = torch.randn(
            B, frames, H*W, D,
            device=ModelArgs().device, dtype=ModelArgs().dtype
        )
        padding_mask_new = torch.randint(
            0, 2, (B, frames),
            device=ModelArgs().device, dtype=torch.bool
        )
        out = attn(x_new, H, W, frames, padding_mask=padding_mask_new)
        assert out.shape == (B, frames, H*W, D)
        assert not out.isnan().any()
        assert not out.isinf().any()
        assert not out.norm().isnan().any()
        assert not out.norm().isinf().any()
        assert out.isfinite().all()
        assert out.isreal().all()
        assert out.norm().isfinite().all()
        assert out.norm().isreal().all()

def test_variable_pixels(attn, padding_mask):
    H_new = random.randint(1, 64)
    W_new = random.randint(1, 64)
    x_new = torch.randn(
        B, T_frames, H_new*W_new, D,
        device=ModelArgs().device, dtype=ModelArgs().dtype
    )
    out = attn(x_new, H_new, W_new, T_frames, padding_mask=padding_mask)
    assert out.shape == (B, T_frames, H_new*W_new, D)
    assert not out.isnan().any()
    assert not out.isinf().any()
    assert not out.norm().isnan().any()
    assert not out.norm().isinf().any()
    assert out.isfinite().all()
    assert out.isreal().all()
    assert out.norm().isfinite().all()
    assert out.norm().isreal().all()
    
def test_zero_input_frames(attn):
    frames = 0
    x_new = torch.randn(
        B, frames, H*W, D,
        device=ModelArgs().device, dtype=ModelArgs().dtype
    )
    padding_mask_new = torch.randint(
        0, 2, (B, frames), device=ModelArgs().device, dtype=torch.bool
    )
    out = attn(x_new, H, W, frames, padding_mask=padding_mask_new)
    assert out.shape == (B, frames, H*W, D)
    assert not out.isnan().any()
    assert not out.isinf().any()
    assert not out.norm().isnan().any()
    assert not out.norm().isinf().any()
    assert out.isfinite().all()
    assert out.isreal().all()
    assert out.norm().isfinite().all()
    assert out.norm().isreal().all()

def test_no_qk_norm(attn, x, padding_mask):
    out1 = attn(x, H, W, T_frames, use_qk_norm=False, padding_mask=padding_mask)
    out2 = attn(x, H, W, T_frames, use_qk_norm=False)
    for out in [out1, out2]:
        assert out.shape == (B, T_frames, H*W, D)
        assert not out.isnan().any()
        assert not out.isinf().any()
        assert not out.norm().isnan().any()
        assert not out.norm().isinf().any()
        assert out.isfinite().all()
        assert out.isreal().all()
        assert out.norm().isfinite().all()
        assert out.norm().isreal().all()

def test_no_causal(attn, x, padding_mask):
    out1 = attn(x, H, W, T_frames, is_causal=False, padding_mask=padding_mask)
    out2 = attn(x, H, W, T_frames, is_causal=False)
    for out in [out1, out2]:
        assert out.shape == (B, T_frames, H*W, D)
        assert not out.isnan().any()
        assert not out.isinf().any()
        assert not out.norm().isnan().any()
        assert not out.norm().isinf().any()
        assert out.isfinite().all()
        assert out.isreal().all()
        assert out.norm().isfinite().all()
        assert out.norm().isreal().all()

def test_spatial_qkv_shapes(attn, x, padding_mask):
    _, q, k, v, _, _, _ = attn(
        x, H, W, T_frames, 
        padding_mask=padding_mask, 
        _return_debug=True
    )
    head_dim = D//ModelArgs().num_heads
    for tensor in [q, k, v]:
        assert tensor.shape == (B*T_frames, ModelArgs().num_heads, H*W, head_dim)

def test_temporal_qkv_shapes(attn, x, padding_mask):
    _, _, _, _, q, k, v = attn(
        x, H, W, T_frames, 
        padding_mask=padding_mask,
        _return_debug=True
    )
    head_dim = D//ModelArgs().num_heads
    for tensor in [q, k, v]:
        assert tensor.shape == (B*H*W, ModelArgs().num_heads, T_frames, head_dim)

def test_attn_interleaves(attn, x, padding_mask):
    out1 = attn(
        x, H, W, T_frames,
        attention_interleave="st",
        padding_mask=padding_mask
    )
    out2 = attn(
        x, H, W, T_frames,
        attention_interleave="ts",
        padding_mask=padding_mask
    )
    for out in [out1, out2]:
        assert out.shape == (B, T_frames, H*W, D)
        assert not out.isnan().any()
        assert not out.isinf().any()
        assert not out.norm().isnan().any()
        assert not out.norm().isinf().any()
        assert out.isfinite().all()
        assert out.isreal().all()
        assert out.norm().isfinite().all()
        assert out.norm().isreal().all()
