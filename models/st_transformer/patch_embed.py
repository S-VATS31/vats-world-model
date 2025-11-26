from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast

class PatchEmbed3D(nn.Module):
    """Convert a video into flattened 3D patch embeddings.

    Args:
        C_in (int): Number of input channels.
        d_model (int): Dimensionality of output patch embeddings.
        patch_size (Tuple[int, int, int]): (pt, ph, pw) patch size for T, H, W.
        use_proj_bias (bool): Whether Conv3d should have bias.
        device (torch.device): Device for parameters.
        dtype (torch.dtype): Data type for parameters.
    """
    def __init__(
        self,
        C_in: int,
        d_model: int,
        patch_size: Tuple[int, int, int],
        use_proj_bias: bool,
        device: torch.device,
        dtype: torch.dtype
    ):
        super().__init__()

        self.patch_size = patch_size # (pt, ph, pw)
        self.d_model = d_model

        self.proj = nn.Conv3d(
            C_in,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
            bias=use_proj_bias,
            device=device,
            dtype=dtype
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int, int], Tensor]:
        """Forward pass.

        Args:
            x (Tensor): Input tensor of shape [B, C, T, H, W].

        Returns:
            Tuple:
                - Tensor: [B, T, Hp*Wp, d_model] patch embeddings
                - Tuple: THW patches as (Tp, Hp, Wp).
                - Tensor: Mask tensor of shape [B, Tp].
        """
        with autocast(device_type=x.device.type):
            B, _, T, H, W = x.shape
            pt, ph, pw = self.patch_size

            # frame mask: False = real frame, True = padded
            frame_mask = torch.zeros((B, T), device=x.device, dtype=torch.bool)

            pad_t = (pt - T % pt) % pt
            pad_h = (ph - H % ph) % ph
            pad_w = (pw - W % pw) % pw

            if pad_t > 0 or pad_h > 0 or pad_w > 0:
                x = nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))
                frame_mask = nn.functional.pad(frame_mask, (0, pad_t), value=True)

            x = self.proj(x) # [B, d_model, Tp, Hp, Wp]

            _, _, Tp, Hp, Wp = x.shape # get T, H, W, patches

            # convert frame mask [B, T+pad_t] to patch mask [B, Tp]
            frames_padded = frame_mask[:, :(Tp * pt)].view(B, Tp, pt)
            patch_mask = frames_padded.any(dim=-1)

            # flatten to [B, Tp, Hp*Wp, d_model]
            x = x.view(B, self.d_model, Tp, -1).permute(0, 2, 3, 1).contiguous()

            return x, (Tp, Hp, Wp), patch_mask
