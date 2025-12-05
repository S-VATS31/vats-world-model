import pytest
import torch
import random

from configs.s_transformer.model_args.xsmall import ModelArgs
from models.s_transformer.patch_embed import PatchEmbed

B, H, W, D = 8, 12, 32, ModelArgs().d_model
ph, pw = ModelArgs().patch_size
num_spatial_patches = H*W

@pytest.fixture
def patch_embed():
    return PatchEmbed(
        C_in=ModelArgs().C_in,
        d_model=D,
        patch_size=ModelArgs().patch_size,
        use_proj_bias=ModelArgs().use_conv2d_bias,
        device=ModelArgs().device,
        dtype=ModelArgs().dtype
    )

@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(
        B, ModelArgs().C_in, H, W,
        device=ModelArgs().device, dtype=ModelArgs().dtype
    )

def test_shape(patch_embed, x):
    out, _, _ = patch_embed(x)
    num_patches_h = (H + ph - 1) // ph
    num_patches_w = (W + pw - 1) // pw
    expected_patches = num_patches_h * num_patches_w
    assert out.shape == (B, expected_patches, D)

def test_return_patches(patch_embed, x):
    _, HW_patches, _ = patch_embed(x)
    H_patches, W_patches = HW_patches
    num_patches_h = (H + ph - 1) // ph
    num_patches_w = (W + pw - 1) // pw
    assert H_patches == num_patches_h
    assert W_patches == num_patches_w
    
def test_mask_shape(patch_embed, x):
    _, _, mask = patch_embed(x)
    num_patches_h = (H + ph - 1) // ph
    num_patches_w = (W + pw - 1) // pw
    assert mask.shape == (B, num_patches_h*num_patches_w)

def test_gradient_flow(patch_embed, x):
    out, _, _ = patch_embed(x)
    loss = out.sum()
    loss.backward()
    for _, param in patch_embed.named_parameters():
        assert torch.isfinite(param.grad).all()
        assert torch.isreal(param.grad).all()
        assert not torch.isnan(param.grad).any()

def test_numerical_stability(patch_embed, x):
    out, _, _ = patch_embed(x)
    assert not out.isnan().any()
    assert not out.isinf().any()
    assert not out.norm().isnan().any()
    assert not out.norm().isinf().any()
    assert out.isfinite().all()
    assert out.isreal().all()
    assert out.norm().isfinite().all()
    assert out.norm().isreal().all()

def test_variable_batch_size(patch_embed):
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        x_new = torch.randn(
            batch_size, ModelArgs().C_in, H, W,
            device=ModelArgs().device, dtype=ModelArgs().dtype
        )
        out, _, mask = patch_embed(x_new)
        num_patches_h = (H + ph - 1) // ph
        num_patches_w = (W + pw - 1) // pw
        expected_patches = num_patches_h * num_patches_w
        assert out.shape == (batch_size, expected_patches, D)
        assert mask.shape == (batch_size, expected_patches)

def test_variable_input_patch(patch_embed):
    H_new = random.randint(1, 64)
    W_new = random.randint(1, 64)
    x_new = torch.randn(
        B, ModelArgs().C_in, H_new, W_new,
        device=ModelArgs().device, dtype=ModelArgs().dtype
    )
    out, _, mask = patch_embed(x_new)
    num_patches_h = (H_new + ph - 1) // ph
    num_patches_w = (W_new + pw  - 1) // pw
    expected_patches = num_patches_h * num_patches_w
    assert out.shape == (B, expected_patches, D)
    assert mask.shape == (B, expected_patches)

def test_deterministic_output(patch_embed, x):
    patch_embed.eval()
    out1, _, _ = patch_embed(x)
    out2, _, _ = patch_embed(x)
    torch.testing.assert_close(out1, out2)
