import pytest
import torch

from configs.text_transformer.model_args.xsmall import ModelArgs
from models.text_transformer.attention import CausalAttention

B, T_tokens, D = 10, 15, ModelArgs().d_model

@pytest.fixture
def model_args():
    return ModelArgs()

@pytest.fixture
def attn(model_args):
    return CausalAttention(
        d_model=model_args.d_model,
        num_heads=model_args.num_heads,
        query_groups=model_args.query_groups,
        use_qkv_bias=model_args.use_qkv_bias,
        use_o_bias=model_args.use_o_bias,
        use_qkv_proj=model_args.use_qkv_proj,
        rope_theta=model_args.rope_theta,
        softmax_scale=model_args.softmax_scale,
        device=model_args.device,
        dtype=model_args.dtype
    )

@pytest.fixture
def x(model_args):
    torch.manual_seed(0)
    return torch.randn(
        B, T_tokens, D,
        device=model_args.device, dtype=model_args.dtype
    )

@pytest.fixture
def padding_mask(model_args):
    torch.manual_seed(0)
    return torch.randint(
        0, 2, (B, T_tokens), device=model_args.device, dtype=torch.bool
    )

def test_shape(attn, x, padding_mask):
    out1 = attn(x, padding_mask=padding_mask)
    out2 = attn(x)
    for out in [out1, out2]:
        assert out.shape == (B, T_tokens, D)

def test_gradient_flow(attn, x, padding_mask):
    out1 = attn(x, padding_mask=padding_mask)
    out2 = attn(x)
    for out in [out1, out2]:
        loss = out.sum()
        loss.backward()
        for _, param in attn.named_parameters():
            assert torch.isfinite(param.grad).all()
            assert torch.isreal(param.grad).all()
            assert not torch.isnan(param.grad).any()    

def test_numerical_stability(attn, x, padding_mask):
    out1 = attn(x, padding_mask=padding_mask)
    out2 = attn(x)
    for out in [out1, out2]:
        assert out.shape == (B, T_tokens, D)
        assert not out.isnan().any()
        assert not out.isinf().any()
        assert not out.norm().isnan().any()
        assert not out.norm().isinf().any()
        assert out.isfinite().all()
        assert out.isreal().all()
        assert out.norm().isfinite().all()
        assert out.norm().isreal().all()    

def test_deterministic_output(attn, x, padding_mask):
    attn.eval()
    out1 = attn(x)
    out2 = attn(x)
    out3 = attn(x, padding_mask=padding_mask)
    out4 = attn(x, padding_mask=padding_mask)
    torch.testing.assert_close(out1, out2)
    torch.testing.assert_close(out3, out4)

def test_variable_batch_size(attn):
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        x_new = torch.randn(
            batch_size, T_tokens, D,
            device=ModelArgs().device, dtype=ModelArgs().dtype
        )
        padding_mask_new = torch.randint(
            0, 2, (batch_size, T_tokens),
            device=ModelArgs().device, dtype=torch.bool
        )
        out = attn(x_new, padding_mask=padding_mask_new)
        assert out.shape == (batch_size, T_tokens, D)
        assert not out.isnan().any()
        assert not out.isinf().any()
        assert not out.norm().isnan().any()
        assert not out.norm().isinf().any()
        assert out.isfinite().all()
        assert out.isreal().all()
        assert out.norm().isfinite().all()
        assert out.norm().isreal().all()
        

def test_variable_input_tokens(attn):
    for tokens in range(1, 100):
        x_new = torch.randn(
            B, tokens, D, device=ModelArgs().device, dtype=ModelArgs().dtype
        )
        padding_mask_new = torch.randint(
            0, 2, (B, tokens), device=ModelArgs().device, dtype=torch.bool
        )
        out = attn(x_new, padding_mask=padding_mask_new)
        assert out.shape == (B, tokens, D)
        assert not out.isnan().any()
        assert not out.isinf().any()
        assert not out.norm().isnan().any()
        assert not out.norm().isinf().any()
        assert out.isfinite().all()
        assert out.isreal().all()
        assert out.norm().isfinite().all()
        assert out.norm().isreal().all()

def test_zero_input_tokens(attn):
    tokens = 0
    x_new = torch.randn(
        B, tokens, D, device=ModelArgs().device, dtype=ModelArgs().dtype
    )
    padding_mask_new = torch.randint(
        0, 2, (B, tokens), device=ModelArgs().device, dtype=torch.bool
    )
    out = attn(x_new, padding_mask=padding_mask_new)
    assert out.shape == (B, tokens, D)
    assert not out.isnan().any()
    assert not out.isinf().any()
    assert not out.norm().isnan().any()
    assert not out.norm().isinf().any()
    assert out.isfinite().all()
    assert out.isreal().all()
    assert out.norm().isfinite().all()
    assert out.norm().isreal().all()
