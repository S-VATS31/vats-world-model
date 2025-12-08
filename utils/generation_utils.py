import torch
from torch import Tensor

def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """Apply temperature scaling to input logits.
    
    Args:
        logits (Tensor): Logits tensor.
        temperature (float): Scaling factor.

    Returns:
        Tensor: Returns scaled logits tensor.
    """
    assert temperature > 0
    return logits / temperature

def apply_top_k(logits: Tensor, top_k_val: int) -> Tensor:
    """Apply top-k sampling to input logits.
    
    Args:
        logits (Tensor): Logits tensor.
        top_k_val (int): Number of logits to sample.
    
    Returns:
        Tensor: Returns logits with top-k selected logits.
    """
    assert top_k_val > 0
    values, indices = torch.topk(logits, top_k_val)
    mask = torch.full_like(logits, float('-inf'))
    mask.scatter_(0, indices, values)
    return mask

def apply_top_p(logits: Tensor, top_p_val: int) -> Tensor:
    """Apply top-p sampling to input logits.
    
    Args:
        logits (Tensor): Logits tensor.
        top_p_val (int): Threshold for top-p sampling.

    Returns:
        Tensor: Returns logits with top-p sampling applied.
    """
    assert 0 < top_p_val <= 1
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)
    mask = cumulative_probs > top_p_val
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits[mask] = float('-inf')
    logits[...] = float('-inf')
    logits.scatter_(0, sorted_indices, sorted_logits)
    return logits

def apply_typical_p(logits: Tensor, typical_p_val: float) -> Tensor:
    """Apply typical-p sampling to input logits.
    
    Args:
        logits (Tensor): Logits tensor.
        typical_p_val (float): Threshold for typical-p sampling.

    Returns:
        Tensor: Returns logits with typical-p sampling applied.
    """
    assert 0 < typical_p_val <= 1
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs)
    neg_log_probs = -log_probs
    entropy = (probs * neg_log_probs).sum()
    distances = torch.abs(neg_log_probs - entropy)
    _, sorted_indices = torch.sort(distances, descending=False)
    sorted_probs = probs[sorted_indices]
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs > typical_p_val
    mask[..., 0] = False
    logits[...] = float('-inf')
    logits.scatter_(0, sorted_indices, logits[sorted_indices].masked_fill(mask, float('-inf')))
    return logits
