import pytest
import torch
import random

from configs.st_transformer.model_args.xsmall import ModelArgs
from models.st_transformer.st_block import SpatioTemporalTransformerBlock

B, T_frames, H, W, D = 20, 15, 12, 32, ModelArgs().d_model

@pytest.fixture
def model_args():
    return ModelArgs()

@pytest.fixture
def block(model_args):
    return SpatioTemporalTransformerBlock(
        d_model=model_args.d_model,
        num_heads=model_args.num_heads,
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

def test_shape(block, x, padding_mask):
    out1 = block(x, H, W, T_frames, padding_mask=padding_mask)
    out2 = block(x, H, W, T_frames)
    for out in [out1, out2]:
        assert out.shape == (B, T_frames, H*W, D)

def test_gradient_flow(block, x, padding_mask):
    out1 = block(x, H, W, T_frames, padding_mask=padding_mask)
    out2 = block(x, H, W, T_frames)
    for out in [out1, out2]:
        loss = out.sum()
        loss.backward()
        for _, param in block.named_parameters():
            assert torch.isfinite(param.grad).all()
            assert torch.isreal(param.grad).all()
            assert not torch.isnan(param.grad).any()

def test_numerical_stability(block, x, padding_mask):
    out1 = block(x, H, W, T_frames, padding_mask=padding_mask)
    out2 = block(x, H, W, T_frames)
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

def test_deterministic_output(block, x, padding_mask):
    block.eval()
    out1_mask = block(x, H, W, T_frames, padding_mask=padding_mask)
    out2_mask = block(x, H, W, T_frames, padding_mask=padding_mask)
    out1_no_mask = block(x, H, W, T_frames)
    out2_no_mask = block(x, H, W, T_frames)
    torch.testing.assert_close(out1_mask, out2_mask)
    torch.testing.assert_close(out1_no_mask, out2_no_mask)

def test_non_deterministic_output(block, x, padding_mask):
    block.train()
    out1_mask = block(x, H, W, T_frames, padding_mask=padding_mask)
    out2_mask = block(x, H, W, T_frames, padding_mask=padding_mask)
    out1_no_mask = block(x, H, W, T_frames)
    out2_no_mask = block(x, H, W, T_frames)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(out1_mask, out2_mask)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(out1_no_mask, out2_no_mask)

def test_variable_batch_sizes(block):
    for batch_size in [1, 2, 4, 8, 16, 32, 64]:
        x_new = torch.randn(
            batch_size, T_frames, H*W, D,
            device=ModelArgs().device, dtype=ModelArgs().dtype
        )
        padding_mask_new = torch.randint(
            0, 2, (batch_size, T_frames),
            device=ModelArgs().device, dtype=ModelArgs().dtype
        )
        out = block(x_new, H, W, T_frames, padding_mask=padding_mask_new)
        assert out.shape == (batch_size, T_frames, H*W, D)
        assert not out.isnan().any()
        assert not out.isinf().any()
        assert not out.norm().isnan().any()
        assert not out.norm().isinf().any()
        assert out.isfinite().all()
        assert out.isreal().all()
        assert out.norm().isfinite().all()
        assert out.norm().isreal().all()

def test_zero_input_frames(block):
    frames = 0
    x_new = torch.randn(
        B, frames, H*W, D,
        device=ModelArgs().device, dtype=ModelArgs().dtype
    )
    padding_mask_new = torch.randint(
        0, 2, (B, frames), 
        device=ModelArgs().device, dtype=ModelArgs().dtype
    )
    out = block(x_new, H, W, frames, padding_mask=padding_mask_new)
    assert out.shape == (B, frames, H*W, D)
    assert not out.isnan().any()
    assert not out.isinf().any()
    assert not out.norm().isnan().any()
    assert not out.norm().isinf().any()
    assert out.isfinite().all()
    assert out.isreal().all()
    assert out.norm().isfinite().all()
    assert out.norm().isreal().all()

def test_variable_pixels(block, padding_mask):
    H_new = random.randint(1, 64)
    W_new = random.randint(1, 64)
    x_new = torch.randn(
        B, T_frames, H_new*W_new, D,
        device=ModelArgs().device, dtype=ModelArgs().dtype
    )
    out = block(x_new, H_new, W_new, T_frames, padding_mask=padding_mask)
    assert out.shape == (B, T_frames, H_new*W_new, D)
    assert not out.isnan().any()
    assert not out.isinf().any()
    assert not out.norm().isnan().any()
    assert not out.norm().isinf().any()
    assert out.isfinite().all()
    assert out.isreal().all()
    assert out.norm().isfinite().all()
    assert out.norm().isreal().all()
    