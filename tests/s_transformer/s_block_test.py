import pytest
import torch
import random

from configs.s_transformer.model_args.xsmall import ModelArgs
from models.s_transformer.s_block import SpatialTransformerBlock

B, H, W, D = 8, 12, 32, ModelArgs().d_model
num_spatial_patches = H*W

@pytest.fixture
def block():
    return SpatialTransformerBlock(
        d_model=D,
        num_heads=ModelArgs().num_heads,
        query_groups=ModelArgs().query_groups,
        rope_theta=ModelArgs().rope_theta,
        softmax_scale=ModelArgs().softmax_scale,
        use_qkv_bias=ModelArgs().use_qkv_bias,
        use_o_bias=ModelArgs().use_o_bias,
        use_qkv_proj=ModelArgs().use_qkv_proj,
        rms_norm_eps=ModelArgs().rms_norm_eps,
        dropout_prob=ModelArgs().dropout_prob,
        d_ffn=ModelArgs().d_ffn,
        use_mlp_bias=ModelArgs().use_mlp_bias,
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

def test_shape(block, x, padding_mask):
    out1 = block(x, H, W, padding_mask=padding_mask)
    out2 = block(x, H, W)
    for out in [out1, out2]:
        assert out.shape == (B, num_spatial_patches, D)

def test_no_padding_mask(block, x):
    out = block(x, H, W)
    assert out.shape == (B, num_spatial_patches, D)
    assert not out.isnan().any()
    assert not out.isinf().any()
    assert not out.norm().isnan().any()
    assert not out.norm().isinf().any()
    assert out.isfinite().all()
    assert out.isreal().all()
    assert out.norm().isfinite().all()
    assert out.norm().isreal().all()

def test_gradient_flow(block, x, padding_mask):
    out1 = block(x, H, W, padding_mask=padding_mask)
    out2 = block(x, H, W)
    for out in [out1, out2]:
        loss = out.sum()
        loss.backward()
        for _, param in block.named_parameters():
            assert torch.isfinite(param.grad).all()
            assert torch.isreal(param.grad).all()
            assert not torch.isnan(param.grad).any()

def test_numerical_stability(block, x, padding_mask):
    out1 = block(x, H, W, padding_mask=padding_mask)
    out2 = block(x, H, W)
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

def test_variable_batch_sizes(block):
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        x_new = torch.randn(
            batch_size, num_spatial_patches, D, device=ModelArgs().device, dtype=ModelArgs().dtype
        )
        padding_mask_new = torch.randint(
            0, 2, (batch_size, num_spatial_patches), device=ModelArgs().device, dtype=torch.bool
        )
        out = block(x_new, H, W, padding_mask=padding_mask_new)
        assert out.shape == (batch_size, num_spatial_patches, D)
        assert not out.isnan().any()
        assert not out.isinf().any()
        assert not out.norm().isnan().any()
        assert not out.norm().isinf().any()
        assert out.isfinite().all()
        assert out.isreal().all()
        assert out.norm().isfinite().all()
        assert out.norm().isreal().all()

def test_variable_input_patches(block):
    H_new = random.randint(1, 64)
    W_new = random.randint(1, 64)
    patches_new = H_new*W_new
    x_new = torch.randn(
        B, patches_new, D, device=ModelArgs().device, dtype=ModelArgs().dtype
    )
    padding_mask_new = torch.randint(
        0, 2, (B, patches_new), device=ModelArgs().device, dtype=torch.bool
    )
    out = block(x_new, H_new, W_new, padding_mask=padding_mask_new)
    assert out.shape == (B, patches_new, D)
    assert not out.isnan().any()
    assert not out.isinf().any()
    assert not out.norm().isnan().any()
    assert not out.norm().isinf().any()
    assert out.isfinite().all()
    assert out.isreal().all()
    assert out.norm().isfinite().all()
    assert out.norm().isreal().all()

def test_deterministic_output(block, x, padding_mask):
    block.eval()
    out1_mask = block(x, H, W, padding_mask=padding_mask)
    out2_mask = block(x, H, W, padding_mask=padding_mask)
    out1_no_mask = block(x, H, W)
    out2_no_mask = block(x, H, W)
    torch.testing.assert_close(out1_mask, out2_mask)
    torch.testing.assert_close(out1_no_mask, out2_no_mask)

def test_non_deterministic_output(block, x, padding_mask):
    block.train()
    out1_mask = block(x, H, W, padding_mask=padding_mask)
    out2_mask = block(x, H, W, padding_mask=padding_mask)
    out1_no_mask = block(x, H, W)
    out2_no_mask = block(x, H, W)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(out1_mask, out2_mask)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(out1_no_mask, out2_no_mask)
