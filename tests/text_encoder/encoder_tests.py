import pytest
import torch

from configs.text_encoder.model_args.xsmall import ModelArgs
from models.text_encoder.encoder import TransformerEncoder

B, T, D, V = 16, 8, ModelArgs().d_model, ModelArgs().vocab_size

@pytest.fixture
def model():
    return TransformerEncoder(ModelArgs())

@pytest.fixture
def input_ids():
    torch.manual_seed(0)
    return torch.randint(
        0, V, (B, T), device=ModelArgs().device, dtype=torch.int64
    )

@pytest.fixture
def padding_mask():
    torch.manual_seed(0)
    return torch.randint(
        0, 2, (B, T), device=ModelArgs().device, dtype=torch.bool
    )

def test_output_shape(model, input_ids, padding_mask):
    out = model(input_ids, padding_mask)
    assert out.shape == (B, T, D)

def test_no_padding_mask(model, input_ids, padding_mask):
    out = model(input_ids, None)
    assert out.shape == (B, T, D)

def test_input_ids_shape(model, input_ids, padding_mask):
    assert input_ids.shape == padding_mask.shape
    out = model(input_ids, padding_mask)
    assert out.shape == (B, T, D)

def test_deterministic_output(model, input_ids, padding_mask):
    model.eval()
    out1_mask = model(input_ids, padding_mask)
    out2_mask = model(input_ids, padding_mask)
    out1_no_mask = model(input_ids)
    out2_no_mask = model(input_ids)
    torch.testing.assert_close(out1_mask, out2_mask)
    torch.testing.assert_close(out1_no_mask, out2_no_mask)

def test_non_deterministic_output(model, input_ids, padding_mask):
    out1_mask = model(input_ids, padding_mask)
    out2_mask = model(input_ids, padding_mask)
    out1_no_mask = model(input_ids)
    out2_no_mask = model(input_ids)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(out1_mask, out2_mask)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(out1_no_mask, out2_no_mask)

def test_variable_batch_sizes(model):
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        input_ids_new = torch.randint(
            0, V, (batch_size, T),device=ModelArgs().device, dtype=torch.int64
        )
        padding_mask_new = torch.randint(
            0, 2, (batch_size, T), device=ModelArgs().device, dtype=torch.bool
        )
        out = model(input_ids_new, padding_mask_new)
        assert out.shape == (batch_size, T, D)
        assert out.isreal().all()
        assert out.isfinite().all()
        assert not out.isinf().any()
        assert not out.isnan().any()

def test_variable_seq_len(model):
    for seq_len in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        input_ids_new = torch.randint(
            0, V, (B, seq_len),device=ModelArgs().device, dtype=torch.int64
        )
        padding_mask_new = torch.randint(
            0, 2, (B, seq_len), device=ModelArgs().device, dtype=torch.bool
        )
        out = model(input_ids_new, padding_mask_new)
        assert out.shape == (B, seq_len, D)
        assert out.isreal().all()
        assert out.isfinite().all()
        assert not out.isinf().any()
        assert not out.isnan().any()

def test_numerical_stability(model, input_ids, padding_mask):
    out1 = model(input_ids)
    out2 = model(input_ids, padding_mask)
    for out in [out1, out2]:
        assert out.isreal().all()
        assert out.isfinite().all()
        assert not out.isinf().any()
        assert not out.isnan().any()

def test_gradient_flow(model, input_ids, padding_mask):
    out1 = model(input_ids)
    out2 = model(input_ids, padding_mask)
    for out in [out1, out2]:
        loss = out.sum()
        loss.backward()
        for _, param in model.named_parameters():
            assert param.grad.isreal().all()
            assert param.grad.isfinite().all()
            assert not param.grad.isinf().any()
            assert not param.grad.isnan().any()
