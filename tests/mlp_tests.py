import pytest
import torch

from configs.text_encoder.model_args.xsmall import ModelArgs
from models.mlp import MLP

B, T, D = 16, 8, ModelArgs().d_model

@pytest.fixture
def mlp():
    return MLP(
        d_model=D,
        d_ffn=ModelArgs().d_ffn,
        dropout_prob=ModelArgs().dropout_prob,
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

def test_shape(mlp, x):
    out = mlp(x)
    assert out.shape == (B, T, D)

def test_deterministic_output(mlp, x):
    mlp.eval()
    out1 = mlp(x)
    out2 = mlp(x)
    torch.testing.assert_close(out1, out2)

def test_gradient_flow(mlp, x):
    out = mlp(x)
    loss = out.sum()
    loss.backward()
    for _, param in mlp.named_parameters():
        assert param.grad.isreal().all()
        assert param.grad.isfinite().all()
        assert not param.grad.isinf().any()
        assert not param.grad.isnan().any()

def test_numerical_stability(mlp, x):
    out = mlp(x)
    assert out.isreal().all()
    assert out.isfinite().all()
    assert not out.isinf().any()
    assert not out.isnan().any()

def test_zero_tokens(mlp):
    x_0 = torch.randn(B, 0, D, device=ModelArgs().device, dtype=ModelArgs().dtype)
    out = mlp(x_0)
    assert out.shape == (B, 0, D)
    assert out.isreal().all()
    assert out.isfinite().all()
    assert not out.isinf().any()
    assert not out.isnan().any()
