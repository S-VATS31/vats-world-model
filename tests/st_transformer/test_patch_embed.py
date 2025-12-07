import pytest
import torch
import random

from configs.st_transformer.model_args.xsmall import ModelArgs
from models.st_transformer.patch_embed import PatchEmbed3D

B, T_frames, H, W, D = 20, 15, 12, 32, ModelArgs().d_model
pt, ph, pw = ModelArgs().patch_size

@pytest.fixture
def model_args():
    return ModelArgs()

@pytest.fixture
def patch_embed(model_args):
    return PatchEmbed3D(
        C_in=model_args.C_in,
        d_model=model_args.d_model,
        patch_size=model_args.patch_size,
        use_proj_bias=model_args.use_conv3d_bias,
        device=model_args.device,
        dtype=model_args.dtype
    )

@pytest.fixture
def x(model_args):
    torch.manual_seed(0)
    return torch.randn(
        B, model_args.C_in, T_frames, H, W,
        device=model_args.device, dtype=model_args.dtype
    )

def test_shape(patch_embed, x):
    out, _, _ = patch_embed(x)
    H_p = (H + ph - 1) // ph
    W_p = (W + pw - 1) // pw
    T_p = (T_frames + pt - 1) // pt
    assert out.shape == (B, T_p, H_p*W_p, D)

def test_thw_patches(patch_embed, x):
    _, THW_patches, _ = patch_embed(x)
    H_p = (H + ph - 1) // ph
    W_p = (W + pw - 1) // pw
    T_p = (T_frames + pt - 1) // pt
    assert (T_p, H_p, W_p) == (THW_patches)

def test_mask(patch_embed, x):
    _, _, mask = patch_embed(x)
    T_p = (T_frames + pt - 1) // pt
    assert mask.shape == (B, T_p)

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

def test_deterministic_output(patch_embed, x):
    patch_embed.eval()
    out1, _, _ = patch_embed(x)
    out2, _, _ = patch_embed(x)
    torch.testing.assert_close(out1, out2)

def test_variable_batch_size(patch_embed):
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        x_new = torch.randn(
            batch_size, ModelArgs().C_in, T_frames, H, W,
            device=ModelArgs().device, dtype=ModelArgs().dtype
        )
        out, _, _ = patch_embed(x_new)
        H_p = (H + ph - 1) // ph
        W_p = (W + pw - 1) // pw
        T_p = (T_frames + pt - 1) // pt
        assert out.shape == (batch_size, T_p, H_p*W_p, D)
        assert not out.isnan().any()
        assert not out.isinf().any()
        assert not out.norm().isnan().any()
        assert not out.norm().isinf().any()
        assert out.isfinite().all()
        assert out.isreal().all()
        assert out.norm().isfinite().all()
        assert out.norm().isreal().all()

def test_variable_frames(patch_embed):
    for frames in range(1, 100):
        x_new = torch.randn(
            B, ModelArgs().C_in, frames, H, W,
            device=ModelArgs().device, dtype=ModelArgs().dtype
        )
        out, _, _ = patch_embed(x_new)  
        H_p = (H + ph - 1) // ph
        W_p = (W + pw - 1) // pw
        T_p = (frames + pt - 1) // pt
        assert out.shape == (B, T_p, H_p*W_p, D)
        assert not out.isnan().any()
        assert not out.isinf().any()
        assert not out.norm().isnan().any()
        assert not out.norm().isinf().any()
        assert out.isfinite().all()
        assert out.isreal().all()
        assert out.norm().isfinite().all()
        assert out.norm().isreal().all()

def test_variable_spatial_pixels(patch_embed):
    H_new = random.randint(1, 64)
    W_new = random.randint(1, 64)
    x_new = torch.randn(
        B, ModelArgs().C_in, T_frames, H_new, W_new,
        device=ModelArgs().device, dtype=ModelArgs().dtype
    )
    out, _, _ = patch_embed(x_new)
    H_p = (H_new + ph - 1) // ph
    W_p = (W_new + pw - 1) // pw
    T_p = (T_frames + pt - 1) // pt
    assert out.shape == (B, T_p, H_p*W_p, D)
    assert not out.isnan().any()
    assert not out.isinf().any()
    assert not out.norm().isnan().any()
    assert not out.norm().isinf().any()
    assert out.isfinite().all()
    assert out.isreal().all()
    assert out.norm().isfinite().all()
    assert out.norm().isreal().all()

def test_zero_input_frames(patch_embed):
    frames = 0
    x_new = torch.randn(
        B, ModelArgs().C_in, frames, H, W,
        device=ModelArgs().device, dtype=ModelArgs().dtype
    )
    out, _, _ = patch_embed(x_new)
    H_p = (H + ph - 1) // ph
    W_p = (W + pw - 1) // pw
    assert out.shape == (B, 0, 0, D)
    assert not out.isnan().any()
    assert not out.isinf().any()
    assert not out.norm().isnan().any()
    assert not out.norm().isinf().any()
    assert out.isfinite().all()
    assert out.isreal().all()
    assert out.norm().isfinite().all()
    assert out.norm().isreal().all()

def test_contiguous_out(patch_embed, x):
    out, _, _ = patch_embed(x)
    assert out.is_contiguous()
