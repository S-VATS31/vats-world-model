import pytest
import torch

from configs.text_encoder.model_args.xsmall import ModelArgs
from models.text_encoder.attention import AttentionBlock

B, T, D = 16, 8, ModelArgs().d_model

@pytest.fixture
def attn_block():
    return AttentionBlock(
        num_heads=ModelArgs().num_heads,
        d_model=D,
        query_groups=ModelArgs().query_groups,
        rope_theta=ModelArgs().rope_theta,
        softmax_scale=ModelArgs().rope_theta,
        use_proj_bias=ModelArgs().use_proj_bias,
        use_qkv_proj=ModelArgs().use_qkv_proj,
        rms_norm_eps=ModelArgs().rms_norm_eps,
        dropout_prob=ModelArgs().dropout_prob,
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

def test_shape(attn_block, x, padding_mask):
    out = attn_block(x, True, False, padding_mask=padding_mask)
    assert out.shape == (B, T, D)

def test_no_padding(attn_block, x):
    out = attn_block(x, True, False)
    assert out.shape == (B, T, D)
    assert out.isreal().all()
    assert out.isfinite().all()
    assert not out.isinf().any()
    assert not out.isnan().any()

def test_zero_tokens(attn_block):
    x_0 = torch.randn(B, 0, D, device=ModelArgs().device, dtype=ModelArgs().dtype)
    padding_mask_0 = torch.randint(
        0, 2, (B, 0), device=ModelArgs().device, dtype=torch.bool
    )
    out = attn_block(x_0, True, False, padding_mask=padding_mask_0)
    assert out.shape == (B, 0, D)
    assert out.isreal().all()
    assert out.isfinite().all()
    assert not out.isinf().any()
    assert not out.isnan().any()

def test_gradient_flow(attn_block, x, padding_mask):
    out = attn_block(x, True, False, padding_mask=padding_mask)
    loss = out.sum()
    loss.backward()
    for _, param in attn_block.named_parameters():
        assert torch.isreal(param.grad).all()
        assert torch.isfinite(param.grad).all()
        assert not torch.isnan(param.grad).any() 
        assert not torch.isinf(param.grad).any() 

def test_deterministic_output(attn_block, x, padding_mask):
    attn_block.eval()
    out1 = attn_block(x, True, False, padding_mask=padding_mask)
    out2 = attn_block(x, True, False, padding_mask=padding_mask)
    torch.testing.assert_close(out1, out2)
