import pytest
import torch
import random

from configs.s_transformer.model_args.xsmall import ModelArgs
from models.s_transformer.s_attention import SpatialAttention

B, H, W, D = 8, 12, 32, ModelArgs().d_model
num_spatial_patches = H*W

@pytest.fixture
def attn():
    return SpatialAttention(
        d_model=D,
        num_heads=ModelArgs().num_heads,
        query_groups=ModelArgs().query_groups,
        rope_theta=ModelArgs().rope_theta,
        softmax_scale=ModelArgs().softmax_scale,
        use_qkv_bias=ModelArgs().use_qkv_bias,
        use_o_bias=ModelArgs().use_o_bias,
        use_qkv_proj=ModelArgs().use_qkv_proj,
        device=ModelArgs().device,
        dtype=ModelArgs().dtype
    )

@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(
        B, num_spatial_patches, D, device=ModelArgs().device, dtype=ModelArgs().dtype
    )

@pytest.fixture
def padding_mask():
    torch.manual_seed(0)
    return torch.randint(
        0, 2, (B, num_spatial_patches), device=ModelArgs().device, dtype=torch.bool
    )

def test_shape(attn, x, padding_mask):
    out1 = attn(x, H, W, padding_mask=padding_mask)
    out2 = attn(x, H, W)
    for out in [out1, out2]:
        assert out.shape == (B, num_spatial_patches, D)

def test_gradient_flow(attn, x, padding_mask):
    out1 = attn(x, H, W, padding_mask=padding_mask)
    out2 = attn(x, H, W)
    for out in [out1, out2]:
        loss = out.sum()
        loss.backward()
        for _, param in attn.named_parameters():
            assert torch.isreal(param.grad).all()
            assert torch.isfinite(param.grad).all()
            assert not torch.isnan(param.grad).any()

def test_no_padding_mask(attn, x):
    out = attn(x, H, W)
    assert out.shape == (B, num_spatial_patches, D)
    assert not out.isnan().any()
    assert not out.isinf().any()
    assert not out.norm().isnan().any()
    assert not out.norm().isinf().any()
    assert out.isfinite().all()
    assert out.isreal().all()
    assert out.norm().isfinite().all()
    assert out.norm().isreal().all()

def test_numerical_stability(attn, x, padding_mask):
    out1 = attn(x, H, W, padding_mask=padding_mask)
    out2 = attn(x, H, W)
    for out in [out1, out2]:
        assert out.shape == (B, num_spatial_patches, D)
        assert not out.isnan().any()
        assert not out.isinf().any()
        assert not out.norm().isnan().any()
        assert not out.norm().isinf().any()
        assert out.isfinite().all()
        assert out.isreal().all()
        assert out.norm().isfinite().all()
        assert out.norm().isreal().all()

def test_no_qk_norm_stability(attn, x, padding_mask):
    out = attn(x, H, W, use_qk_norm=False, padding_mask=padding_mask)
    assert out.shape == (B, num_spatial_patches, D)
    assert not out.isnan().any()
    assert not out.isinf().any()
    assert not out.norm().isnan().any()
    assert not out.norm().isinf().any()
    assert out.isfinite().all()
    assert out.isreal().all()
    assert out.norm().isfinite().all()
    assert out.norm().isreal().all()

def test_qkv_shapes(attn, x, padding_mask):
    out, q, k, v = attn(x, H, W, padding_mask=padding_mask, _return_qkv=True)
    assert out.shape == (B, num_spatial_patches, D)
    for tensor in [q, k, v]:
        assert tensor.shape == (
            B, ModelArgs().num_heads, num_spatial_patches, ModelArgs().d_model//ModelArgs().num_heads
        )

def test_variable_batch_sizes(attn):
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        x_new = torch.randn(
            batch_size, num_spatial_patches, D, device=ModelArgs().device, dtype=ModelArgs().dtype
        )
        padding_mask_new = torch.randint(
            0, 2, (batch_size, num_spatial_patches), device=ModelArgs().device, dtype=torch.bool
        )
        out = attn(x_new, H, W, padding_mask=padding_mask_new)
        assert out.shape == (batch_size, num_spatial_patches, D)
        assert not out.isnan().any()
        assert not out.isinf().any()
        assert not out.norm().isnan().any()
        assert not out.norm().isinf().any()
        assert out.isfinite().all()
        assert out.isreal().all()
        assert out.norm().isfinite().all()
        assert out.norm().isreal().all()

def test_variable_input_patches(attn):
    H_new = random.randint(1, 64)
    W_new = random.randint(1, 64)
    patches_new = H_new*W_new
    x_new = torch.randn(
        B, patches_new, D, device=ModelArgs().device, dtype=ModelArgs().dtype
    )
    padding_mask_new = torch.randint(
        0, 2, (B, patches_new), device=ModelArgs().device, dtype=torch.bool
    )
    out = attn(x_new, H_new, W_new, padding_mask=padding_mask_new)
    assert out.shape == (B, patches_new, D)
    assert not out.isnan().any()
    assert not out.isinf().any()
    assert not out.norm().isnan().any()
    assert not out.norm().isinf().any()
    assert out.isfinite().all()
    assert out.isreal().all()
    assert out.norm().isfinite().all()
    assert out.norm().isreal().all()

def test_deterministic_output(attn, x, padding_mask):
    out1_mask = attn(x, H, W, padding_mask=padding_mask)
    out2_mask = attn(x, H, W, padding_mask=padding_mask)
    out1_no_mask = attn(x, H, W)
    out2_no_mask = attn(x, H, W)
    torch.testing.assert_close(out1_mask, out2_mask)
    torch.testing.assert_close(out1_no_mask, out2_no_mask)
