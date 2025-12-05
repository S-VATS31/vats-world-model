import pytest
import torch

from configs.text_encoder.model_args.xsmall import ModelArgs
from models.text_encoder.attention import Attention

B, T, D = 16, 8, ModelArgs().d_model

@pytest.fixture
def attn():
    return Attention(
        num_heads=ModelArgs().num_heads,
        d_model=D,
        query_groups=ModelArgs().query_groups,
        rope_theta=ModelArgs().rope_theta,
        softmax_scale=ModelArgs().rope_theta,
        use_proj_bias=ModelArgs().use_proj_bias,
        use_qkv_proj=ModelArgs().use_qkv_proj,
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

def test_shape(attn, x, padding_mask):
    out = attn(x, True, False, padding_mask=padding_mask)
    assert out.shape == (B, T, D)

def test_numerical_stability(attn, x, padding_mask):
    out = attn(x, True, False, padding_mask=padding_mask)
    assert not out.isnan().any()
    assert not out.isinf().any()
    assert not out.norm().isnan().any()
    assert not out.norm().isinf().any()
    assert out.isfinite().all()
    assert out.isreal().all()
    assert out.norm().isfinite().all()
    assert out.norm().isreal().all()

def test_no_padding(attn, x):
    out = attn(x, True, False)
    assert out.shape == (B, T, D)

def test_gradient_flow(attn, x, padding_mask):
    out = attn(x, True, False, padding_mask=padding_mask)
    loss = out.sum()
    loss.backward()
    for _, param in attn.named_parameters():
        assert torch.isreal(param.grad).all()
        assert torch.isfinite(param.grad).all()
        assert not torch.isnan(param.grad).any() 
        assert not torch.isinf(param.grad).any() 

def test_zero_input_tokens(attn):
    torch.manual_seed(0)
    x_0 = torch.randn(B, 0, D, device=ModelArgs().device, dtype=ModelArgs().dtype)
    padding_mask_0 = torch.randint(
        0, 2, (B, 0), device=ModelArgs().device, dtype=torch.bool
    )
    out = attn(x_0, True, False, padding_mask_0)
    assert out.shape == (B, 0, D)
    assert not out.isnan().any()
    assert not out.isinf().any()
    assert not out.norm().isnan().any()
    assert not out.norm().isinf().any()
    assert out.isfinite().all()
    assert out.isreal().all()
    assert out.norm().isfinite().all()
    assert out.norm().isreal().all()

def test_qkv_shapes(attn, x, padding_mask):
    _, q, k, v = attn(x, True, False, padding_mask=padding_mask, _return_qkv=True)
    correct_shape = (B, ModelArgs().num_heads, T, D/ModelArgs().num_heads)
    assert q.shape == k.shape == v.shape == correct_shape

def test_no_qk_norm(attn, x, padding_mask):
    pass

def test_qk_norm_stability(attn, x, padding_mask):
    pass

def test_no_qk_norm_stability(attn, x, padding_mask):
    pass

def test_mqa(attn, x, padding_mask):
    pass
