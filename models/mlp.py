import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

# TODO: add optimized swiglu if GPUs available

class MLP(nn.Module):
    """MLP network for transformer.
    
    Args:
        d_model (int): Dimensionality of model embeddings.
        d_ffn (int): Dimensionality of MLP.
        dropout_prob (float): Dropout probability for regularization.
        use_mlp_bias (bool): Whether to use MLP bias or not.
        device (torch.device): Accelerator.
        dtype (torch.dtype): Data type of tensors.
    """
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        dropout_prob: float,
        use_mlp_bias: bool,
        device: torch.device,
        dtype: torch.dtype
    ):
        super().__init__()

        self.weight1 = nn.Linear(
            d_model, 
            d_ffn, 
            bias=use_mlp_bias, 
            device=device,
            dtype=dtype
        )
        self.weight2 = nn.Linear(
            d_model,
            d_ffn, 
            bias=use_mlp_bias, 
            device=device, 
            dtype=dtype
        )
        self.weight3 = nn.Linear(
            d_ffn, 
            d_model, 
            bias=use_mlp_bias, 
            device=device, 
            dtype=dtype
        )
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of MLP layer.
        
        Args:
            x (Tensor): Input tensor of shape [B, T, d_model].

        Returns:
            Tensor: Output tensor of same shape.
        """
        with autocast(device_type=x.device.type):
            return self.dropout(self.weight3(F.silu(self.weight1(x)) * self.weight2(x)))

class MLPBlock(nn.Module):
    """MLP Block applying MLP forward, normalization, dropout, and residual.
    
    Args:
        d_model (int): Dimensionality of model embeddings.
        d_ffn (int): Dimensionality of MLP.
        dropout_prob (float): Dropout probability for regularization.
        use_mlp_bias (bool): Whether to use MLP bias or not.
        rms_norm_eps (float): Epsilon value for RMSNorm.
        device (torch.device): Accelerator.
        dtype (torch.dtype): Data type of tensors.
    """
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        dropout_prob: float,
        use_mlp_bias: bool,
        rms_norm_eps: float,
        device: torch.device,
        dtype: torch.dtype
    ):
        super().__init__()

        self.mlp = MLP(
            d_model=d_model,
            d_ffn=d_ffn,
            dropout_prob=dropout_prob,
            use_mlp_bias=use_mlp_bias,
            device=device,
            dtype=dtype
        )
        self.rms_norm = nn.RMSNorm(
            d_model,
            eps=rms_norm_eps,
            device=device,
            dtype=dtype
        )
        self.dropout = nn.Dropout(p=dropout_prob)


    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of MLP Block.
        
        Args:
            x (Tensor): Input tensor of shape [B, T, d_model].

        Returns:
            Tensor: Output tensor of same shape.
        """
        with autocast(device_type=x.device.type):
            return x + self.dropout(self.mlp(self.rms_norm(x)))
