import pytest
import torch

from configs.text_encoder.model_args.xsmall import ModelArgs
from models.text_encoder.block import TransformerBlock

B, T, D = 16, 8, ModelArgs().d_model

@pytest.fixture
def block():
    return TransformerBlock(
        num_heads=ModelArgs().num_heads,
        d_model=ModelArgs().d_model,
        query_groups=ModelArgs().query_groups,
        rope_theta=ModelArgs().rope_theta,
        softmax_scale=ModelArgs().softmax_scale,
        use_proj_bias=ModelArgs().use_proj_bias,
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
        B, T, D, device=ModelArgs().device, dtype=ModelArgs().dtype
    )

@pytest.fixture
def padding_mask():
    torch.manual_seed(0)
    return torch.randint(
        0, 2, (B, T), device=ModelArgs().device, dtype=torch.bool
    )

def test_shape(block, x, padding_mask):
    out = block(x, True, False, padding_mask=padding_mask)
    assert out.shape == (B, T, D)

def test_gradient_flow(block, x, padding_mask):
    out1 = block(x, False, False, padding_mask=None)
    out2 = block(x, False, False, padding_mask=padding_mask)
    out3 = block(x, True, False, padding_mask=None)
    out4 = block(x, True, False, padding_mask=padding_mask)
    for out in [out1, out2, out3, out4]:
        loss = out.sum()
        loss.backward()
        for _, param in block.named_parameters():
            assert param.grad.isreal().all()
            assert param.grad.isfinite().all()
            assert not param.grad.isinf().any()
            assert not param.grad.isnan().any()

def test_numerical_stability(block, x, padding_mask):
    out1 = block(x, False, False, padding_mask=None)
    out2 = block(x, False, False, padding_mask=padding_mask)
    out3 = block(x, True, False, padding_mask=None)
    out4 = block(x, True, False, padding_mask=padding_mask)
    for out in [out1, out2, out3, out4]:
        assert out.isreal().all()
        assert out.isfinite().all()
        assert not out.isnan().any()
        assert not out.isinf().any()

def test_no_padding_mask(block, x):
    out1 = block(x, False, False, padding_mask=None)
    out2 = block(x, True, False, padding_mask=None)
    for out in [out1, out2]:
        assert out.isreal().all()
        assert out.isfinite().all()
        assert not out.isnan().any()
        assert not out.isinf().any()
        assert out.shape == (B, T, D)

def test_mqa(block, x, padding_mask):
    out = block(x, True, True, padding_mask=padding_mask)
    assert out.shape == (B, T, D)

def test_variable_qk_norm_eps(block, x, padding_mask):
    for eps in [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]:
        out = block(x, True, False, eps=eps, padding_mask=padding_mask)
        assert out.isreal().all()
        assert out.isfinite().all()
        assert not out.isnan().any()
        assert not out.isinf().any()
        assert out.shape == (B, T, D)

def test_variable_seq_len(block):
    for seq_len in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        x_new = torch.randn(B, seq_len, D, device=ModelArgs().device, dtype=ModelArgs().dtype)
        padding_mask_new = torch.randint(
            0, 2, (B, seq_len), device=ModelArgs().device, dtype=torch.bool
        )
        out = block(x_new, True, False, padding_mask=padding_mask_new)
        assert out.isreal().all()
        assert out.isfinite().all()
        assert not out.isnan().any()
        assert not out.isinf().any()
        assert out.shape == (B, seq_len, D)

def test_variable_batch_size(block):
    for batch_size in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        x_new = torch.randn(batch_size, T, D, device=ModelArgs().device, dtype=ModelArgs().dtype)
        padding_mask_new = torch.randint(
            0, 2, (batch_size, T), device=ModelArgs().device, dtype=torch.bool
        )
        out = block(x_new, True, False, padding_mask=padding_mask_new)
        assert out.isreal().all()
        assert out.isfinite().all()
        assert not out.isnan().any()
        assert not out.isinf().any()
        assert out.shape == (batch_size, T, D)

def test_deterministic_output(block, x, padding_mask):
    block.eval()
    out1_mask = block(x, True, False, padding_mask=padding_mask)
    out2_mask = block(x, True, False, padding_mask=padding_mask)
    out1_no_mask = block(x, True, False, padding_mask=None)
    out2_no_mask = block(x, True, False, padding_mask=None)
    torch.testing.assert_close(out1_mask, out2_mask)
    torch.testing.assert_close(out1_no_mask, out2_no_mask)

def test_non_deterministic_output(block, x, padding_mask):
    out1_mask = block(x, True, False, padding_mask=padding_mask)
    out2_mask = block(x, True, False, padding_mask=padding_mask)
    out1_no_mask = block(x, True, False, padding_mask=None)
    out2_no_mask = block(x, True, False, padding_mask=None)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(out1_mask, out2_mask)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(out1_no_mask, out2_no_mask)
