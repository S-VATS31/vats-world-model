import pytest
import torch
import random

from configs.s_transformer.model_args.xsmall import ModelArgs
from models.s_transformer.s_transformer import SpatialTransformer

B, H, W, D, V = 8, 12, 32, ModelArgs().d_model, ModelArgs().codebook_size
ph, pw = ModelArgs().patch_size
num_spatial_patches = H*W

@pytest.fixture
def model():
    return SpatialTransformer(ModelArgs())

@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(
        B, ModelArgs().C_in, H, W, device=ModelArgs().device, dtype=ModelArgs().dtype
    )

def test_shape(model, x):
    out = model(x)
    num_patches_h = (H + ph - 1) // ph
    num_patches_w = (W + pw - 1) // pw
    expected_patches = num_patches_h * num_patches_w
    assert out.shape == (B, expected_patches, V)

def test_gradient_flow(model, x):
    out = model(x)
    loss = out.sum()
    loss.backward()
    for _, param in model.named_parameters():
        assert torch.isfinite(param.grad).all()
        assert torch.isreal(param.grad).all()
        assert not torch.isnan(param.grad).any()

def test_numerical_stability(model, x):
    out = model(x)
    assert not out.isnan().any()
    assert not out.isinf().any()
    assert not out.norm().isnan().any()
    assert not out.norm().isinf().any()
    assert out.isfinite().all()
    assert out.isreal().all()
    assert out.norm().isfinite().all()
    assert out.norm().isreal().all()

def test_cache(model, x):
    out = model(x, use_cache=True)
    num_patches_h = (H + ph - 1) // ph
    num_patches_w = (W + pw - 1) // pw
    expected_patches = num_patches_h * num_patches_w
    assert out.shape == (B, expected_patches, V)
    assert not out.isnan().any()
    assert not out.isinf().any()
    assert not out.norm().isnan().any()
    assert not out.norm().isinf().any()
    assert out.isfinite().all()
    assert out.isreal().all()
    assert out.norm().isfinite().all()
    assert out.norm().isreal().all()

def test_variable_batch_sizes(model):
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        x_new = torch.randn(
            batch_size, ModelArgs().C_in, H, W, 
            device=ModelArgs().device, dtype=ModelArgs().dtype
        )
        out = model(x_new)
        num_patches_h = (H + ph - 1) // ph
        num_patches_w = (W + pw - 1) // pw
        expected_patches = num_patches_h * num_patches_w
        assert out.shape == (batch_size, expected_patches, V)
        assert not out.isnan().any()
        assert not out.isinf().any()
        assert not out.norm().isnan().any()
        assert not out.norm().isinf().any()
        assert out.isfinite().all()
        assert out.isreal().all()
        assert out.norm().isfinite().all()
        assert out.norm().isreal().all()

def test_variable_input_patches(model):
    H_new = random.randint(1, 64)
    W_new = random.randint(1, 64)
    x_new = torch.randn(
        B, ModelArgs().C_in, H_new, W_new,
        device=ModelArgs().device, dtype=ModelArgs().dtype
    )
    out = model(x_new)
    num_patches_h = (H_new + ph - 1) // ph
    num_patches_w = (W_new + pw - 1) // pw
    expected_patches = num_patches_h * num_patches_w
    assert out.shape == (B, expected_patches, V)
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
