from torch import Tensor
import torch.nn.functional as F

def extend_kv_heads(
    tensor: Tensor, 
    dim: int, 
    repeats: int, 
    use_mqa: bool = False
) -> Tensor:
    """Extend KV heads for input key or value tensor.
    
    Args:
        tensor (Tensor): Input key or value tensor.
        dim (int): Dimension to be repeated.
        repeats (int): Number of times to repeat dimension.
        use_mqa (bool): Whether to extend heads or not based on MQA usage, defaults False.

    Returns:
        Tensor: Key or value tensor with specific dimension repeated.
    """
    if use_mqa and tensor.size(dim) == 1:
        return tensor
    return tensor.repeat_interleave(repeats, dim=dim)
    
def apply_qk_norm(tensor: Tensor, eps: float = 1e-10, norm: int = 2) -> Tensor:
    """Apply QK normalization to input tensors.
    
    Args:
        tensor (Tensor): Input query or key tensor.
        eps (float): Small value to maintain numerical stability. Defaults to 1e-10.
        norm (int): L_norm normalization. Defaults to L_2 norm.

    Returns:
        Tensor: Normalized query or key tensor.
    """
    return F.normalize(tensor, p=norm, eps=eps)
