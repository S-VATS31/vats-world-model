import math
import torch
from dataclasses import dataclass
from typing import Tuple, Literal

@dataclass
class ModelArgs:
    """Extra small configuration of model arguments."""
    patch_size: Tuple[int, int] = (2, 8, 8)
    C_in: int = 3
    max_frames: int = 512
    d_model: int = 96
    num_heads: int = 4
    query_groups: int = 2
    d_ffn: int = 1024
    num_layers: int = 8
    dropout_prob: float = 0.1
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-7
    qk_norm_eps: float = 1e-8
    gradient_checkpointing: bool = True
    use_qkv_bias: bool = False
    use_o_bias: bool = False
    use_qkv_proj: bool = True
    use_mqa: bool = False
    softmax_scale: float = math.sqrt(256//16)
    use_mlp_bias: bool = False
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32
    use_qk_norm: bool = True
    qk_norm_type: int = 2
    use_conv3d_bias: bool = False
    is_causal: bool = True
    codebook_size: int = 2048
    use_patch_prediction_bias: bool = False
    max_batch_size: int = 512
    attention_interleave: Literal["st", "ts"] = "st"
