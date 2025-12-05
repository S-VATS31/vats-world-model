import pytest
import torch
import random

from configs.st_transformer.model_args.xsmall import ModelArgs
from models.st_transformer.st_transformer import SpatioTemporalTransformer

B, T_frames, H, W, V = 20, 15, 12, 32, ModelArgs().codebook_size
pt, ph, pw = ModelArgs().patch_size

@pytest.fixture
def model_args():
    return ModelArgs()

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

def test_shape(model, x):
    out = model(x)
    T_p = (T_frames + pt - 1) // pt
    H_p = (H + ph - 1) // ph
    W_p = (W + pw - 1) // pw
    assert out.shape == (B, T_p, H_p*W_p, V)

def test_cache(model, x):
    out = model(x, True)
    T_p = (T_frames + pt - 1) // pt
    H_p = (H + ph - 1) // ph
    W_p = (W + pw - 1) // pw
    assert out.shape == (B, T_p, H_p*W_p, V)

def test_gradient_flow(model, x):
    out1 = model(x)
    out2 = model(x, True)
    for out in [out1, out2]:
        loss = out.sum()
        loss.backward()
        for _, param in model.named_parameters():
            assert torch.isfinite(param.grad).all()
            assert torch.isreal(param.grad).all()
            assert not torch.isnan(param.grad).any()

def test_numerical_stability(model, x):
    out1 = model(x)
    out2 = model(x, True)
    for out in [out1, out2]:
        assert not out.isnan().any()
        assert not out.isinf().any()
        assert not out.norm().isnan().any()
        assert not out.norm().isinf().any()
        assert out.isfinite().all()
        assert out.isreal().all()
        assert out.norm().isfinite().all()
        assert out.norm().isreal().all()
        
def test_deterministic_output(model, x):
    model.eval()
    out1 = model(x)
    out2 = model(x)
    torch.testing.assert_close(out1, out2)

def test_non_deterministic_output(model, x):
    model.train()
    out1 = model(x)
    out2 = model(x)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(out1, out2)

def test_variable_batch_sizes(model):
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        x_new = torch.randn(
            batch_size, ModelArgs().C_in, T_frames, H, W,
            device=ModelArgs().device, dtype=ModelArgs().dtype
        )
        out = model(x_new)
        assert not out.isnan().any()
        assert not out.isinf().any()
        assert not out.norm().isnan().any()
        assert not out.norm().isinf().any()
        assert out.isfinite().all()
        assert out.isreal().all()
        assert out.norm().isfinite().all()
        assert out.norm().isreal().all()

def test_variable_frames(model):
    for frames in range(1, 100):
        x_new = torch.randn(
            B, ModelArgs().C_in, frames, H, W,
            device=ModelArgs().device, dtype=ModelArgs().dtype
        )
        out = model(x_new)
        assert not out.isnan().any()
        assert not out.isinf().any()
        assert not out.norm().isnan().any()
        assert not out.norm().isinf().any()
        assert out.isfinite().all()
        assert out.isreal().all()
        assert out.norm().isfinite().all()
        assert out.norm().isreal().all()

def test_variable_pixels(model):
    H_new = random.randint(1, 64)
    W_new = random.randint(1, 64)
    x_new = torch.randn(
        B, ModelArgs().C_in, T_frames, H_new, W_new,
        device=ModelArgs().device, dtype=ModelArgs().dtype
    )
    out = model(x_new)
    assert not out.isnan().any()
    assert not out.isinf().any()
    assert not out.norm().isnan().any()
    assert not out.norm().isinf().any()
    assert out.isfinite().all()
    assert out.isreal().all()
    assert out.norm().isfinite().all()
    assert out.norm().isreal().all()

def test_zero_input_frames(model):
    frames = 0
    x_new = torch.randn(
        B, ModelArgs().C_in, frames, H, W,
        device=ModelArgs().device, dtype=ModelArgs().dtype
    )
    out = model(x_new)
    assert not out.isnan().any()
    assert not out.isinf().any()
    assert not out.norm().isnan().any()
    assert not out.norm().isinf().any()
    assert out.isfinite().all()
    assert out.isreal().all()
    assert out.norm().isfinite().all()
    assert out.norm().isreal().all()
