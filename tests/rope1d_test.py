import pytest
import torch

from configs.text_transformer.model_args.xsmall import ModelArgs
from models.text_transformer.attention import RoPE

B, T_tokens, num_heads = 10, 15, ModelArgs().num_heads
head_dim = ModelArgs().d_model // num_heads

@pytest.fixture
def model_args():
    return ModelArgs()

@pytest.fixture
def rope(model_args):
    return RoPE(
        head_dim=head_dim,
        rope_theta=model_args.rope_theta,
        device=model_args.device,
        dtype=model_args.dtype
    )

@pytest.fixture
def x(model_args):
    torch.manual_seed(0)
    return torch.randn(
        B, T_tokens, num_heads, head_dim,
        device=model_args.device, dtype=model_args.dtype
    )

def test_shape(rope, x):
    out = rope(x)
    assert out.shape == (x.shape) == (B, T_tokens, num_heads, head_dim)

def test_numerical_stability(rope, x):
    out = rope(x)
    assert out.isreal().all()
    assert out.isfinite().all()
    assert not out.isinf().any()
    assert not out.isnan().any()

def test_zero_input_tokens(rope):
    x = torch.randn(
        B, 0, num_heads, head_dim, 
        device=ModelArgs().device, dtype=ModelArgs().dtype
    )
    out = rope(x)
    assert out.shape == (B, 0, num_heads, head_dim)
