from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast

class PatchEmbed(nn.Module):
    """Convert an input image into flattened patch embeddings.

    Args:
        C_in (int): Number of input channels (Ex. 3 for RGB).
        d_model (int): Dimensionality of output patch embeddings.
        patch_size (Tuple[int, int]): Height and width of each patch.
        use_proj_bias (bool): Whether to use projection bias for Conv2d.
        device (torch.device): Device for module parameters.
        dtype (torch.dtype): Data type for module parameters.
    """
    def __init__(
        self,
        C_in: int,
        d_model: int,
        patch_size: Tuple[int, int],
        use_proj_bias: bool,
        device: torch.device,
        dtype: torch.dtype
    ):
        super().__init__()

        self.patch_size = patch_size
        self.d_model = d_model

        self.proj = nn.Conv2d(
            C_in,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
            bias=use_proj_bias,
            device=device,
            dtype=dtype
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        """Forward pass of patch embeddings layer.

        Args:
            x (Tensor): Input image tensor of shape [B, C, H, W].

        Returns:
            Tuple[Tensor, Tuple[int, int]]:
                - Tensor: Output tensor of shape [B, Hp*Wp, d_model].
                - Tuple[int, int]: Height and width patch.
        """
        with autocast(device_type=x.device.type):
            _, _, H, W = x.shape
            ph, pw = self.patch_size

            # Compute necessary padding
            pad_h = (ph - H % ph) % ph
            pad_w = (pw - W % pw) % pw

            if pad_h > 0 or pad_w > 0:
                x = nn.functional.pad(x, (0, pad_w, 0, pad_h))

            x = self.proj(x) # [B, d_model, Hp, Wp]
            Hp, Wp = x.size(-2), x.size(-1)
            x = x.view(x.size(0), self.d_model, -1).transpose(1, 2).contiguous()

            return x, (Hp, Wp)
        
def test_patch_embed():
    patch_embed = PatchEmbed(
        C_in=3,
        d_model=512,
        patch_size=(8, 8),
        use_proj_bias=False,
        device="cpu",
        dtype=torch.float32
    )
    B, H, W = 16, 64, 64
    x = torch.randn(B, 3, H, W, device="cpu", dtype=torch.float32)
    out = patch_embed(x)

    return out

if __name__ == "__main__":
    out = test_patch_embed()
    print(out.shape)
