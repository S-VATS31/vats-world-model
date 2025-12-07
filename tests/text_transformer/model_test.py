import pytest
import torch

from configs.text_transformer.model_args.xsmall import ModelArgs
from models.text_transformer.text_transformer import CausalTransformer

B, T_tokens, V = 15, 20, ModelArgs().vocab_size

@pytest.fixture
def model_args():
    return ModelArgs()

@pytest.fixture
def model(model_args):
    return CausalTransformer(model_args)

@pytest.fixture
def input_ids():
    torch.manual_seed(0)
    return torch.randint(
        0, V, (B, T_tokens), device=ModelArgs().device, dtype=torch.int64
    )

@pytest.fixture
def padding_mask():
    torch.manual_seed(0)
    return torch.randint(
        0, 2, (B, T_tokens), device=ModelArgs().device, dtype=ModelArgs().dtype
    )

def test_shape(model, input_ids, padding_mask):
    out1 = model(input_ids)
    out2 = model(input_ids, padding_mask=padding_mask)
    for out in [out1, out2]:
        assert out.shape == (B, T_tokens, V)

def test_gradient_flow(model, input_ids, padding_mask):
    out1 = model(input_ids)
    out2 = model(input_ids, padding_mask=padding_mask)
    for out in [out1, out2]:
        loss = out.sum()
        loss.backward()
        for _, param in model.named_parameters():
            assert torch.isfinite(param.grad).all()
            assert torch.isreal(param.grad).all()
            assert not torch.isnan(param.grad).any()

def test_numerical_stability(model, input_ids, padding_mask):
    out1 = model(input_ids)
    out2 = model(input_ids, padding_mask=padding_mask)
    for out in [out1, out2]:
        assert not out.isnan().any()
        assert not out.isinf().any()
        assert not out.norm().isnan().any()
        assert not out.norm().isinf().any()
        assert out.isfinite().all()
        assert out.isreal().all()
        assert out.norm().isfinite().all()
        assert out.norm().isreal().all()

def test_zero_input_tokens(model):
    tokens = 0
    input_ids_new = torch.randint(
        0, V, (B, tokens), device=ModelArgs().device, dtype=torch.int64
    )
    padding_mask_new = torch.randint(
        0, V, (B, tokens), device=ModelArgs().device, dtype=torch.int64
    )
    out = model(input_ids_new, padding_mask=padding_mask_new)
    assert out.shape == (B, 0, V)
    assert not out.isnan().any()
    assert not out.isinf().any()
    assert not out.norm().isnan().any()
    assert not out.norm().isinf().any()
    assert out.isfinite().all()
    assert out.isreal().all()
    assert out.norm().isfinite().all()
    assert out.norm().isreal().all()
