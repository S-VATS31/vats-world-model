import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint

from configs.s_transformer.model_args.xsmall import ModelArgs
from models.s_transformer.patch_embed import PatchEmbed
from models.s_transformer.spatial_kv_cache import SpatialKVCache
from models.s_transformer.s_block import SpatialTransformerBlock

class SpatialTransformer(nn.Module):
    """Spatial transformer module.
    
    Args:
        model_args (ModelArgs): Model parameters.
    """
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        
        self.model_args = model_args

        # Set up patch embeddings
        self.patch_embed = PatchEmbed(
            C_in=model_args.C_in,
            d_model=model_args.d_model,
            patch_size=model_args.patch_size,
            use_proj_bias=model_args.use_conv2d_bias,
            device=model_args.device,
            dtype=model_args.dtype
        )

        # Set up layers
        self.layers = nn.ModuleList([
            SpatialTransformerBlock(
                d_model=model_args.d_model,
                num_heads=model_args.num_heads,
                query_groups=model_args.query_groups,
                rope_theta=model_args.rope_theta,
                softmax_scale=model_args.softmax_scale,
                use_qkv_bias=model_args.use_qkv_bias,
                use_o_bias=model_args.use_o_bias,
                use_qkv_proj=model_args.use_qkv_proj,
                rms_norm_eps=model_args.rms_norm_eps,
                dropout_prob=model_args.dropout_prob,
                d_ffn=model_args.d_ffn,
                use_mlp_bias=model_args.use_mlp_bias,
                device=model_args.device,
                dtype=model_args.dtype
            ) for _ in range(model_args.num_layers)
        ])

        # Set up KV cache
        self.kv_cache = SpatialKVCache(
            num_heads=model_args.num_heads,
            head_dim=model_args.d_model//model_args.num_heads,
            max_batch_size=model_args.max_batch_size,
            max_patches=model_args.max_patches,
            num_layers=model_args.num_layers,
            device=model_args.device,
            dtype=model_args.dtype
        )

        # Set up dropout
        self.dropout = nn.Dropout(p=model_args.dropout_prob)

        # Set up RMSNorm
        self.rms_norm = nn.RMSNorm(
            model_args.d_model,
            eps=model_args.rms_norm_eps,
            device=model_args.device,
            dtype=model_args.dtype
        )

        # Set up patch prediction head
        self.prediction_head = nn.Linear(
            model_args.d_model,
            self.model_args.codebook_size,
            bias=model_args.use_patch_prediction_bias,
            device=model_args.device,
            dtype=model_args.dtype
        )

        self.apply(self._init_weights)
    
    def _init_weights(self, module) -> None:
        """Initialize weights for linear, conv, and RMSNorm layers.

        Args:
            module: Module to be initialized.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, x: Tensor, use_cache: bool = False) -> Tensor:
        """Forward pass of transformer layer.
        
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W].
            use_cache (bool): Whether to use KV caching or not.
        
        Returns:
            Tensor: Output tensor of shape [B, H*W, codebook_size].
        """
        with autocast(device_type=x.device.type):
            # Get x, patches, and mask from PE
            x, HW_patches, padding_mask = self.patch_embed(x)
            x = self.dropout(x)
            H_patch, W_patch = HW_patches

            # Loop through layers
            for layer_idx, layer in enumerate(self.layers):
                if self.model_args.gradient_checkpointing:
                    x = checkpoint(
                        layer,
                        x,
                        H_patch,
                        W_patch,
                        self.model_args.use_qk_norm,
                        self.model_args.use_mqa,
                        self.model_args.qk_norm_eps,
                        self.model_args.qk_norm_type,
                        use_cache,
                        layer_idx,
                        self.kv_cache,
                        self.model_args.is_causal,
                        padding_mask,
                        use_reentrant=False
                    )
                else:
                    x = layer(
                        x,
                        H_patch=H_patch,
                        W_patch=W_patch,
                        use_qk_norm=self.model_args.use_qk_norm,
                        use_mqa=self.model_args.use_mqa,
                        qk_norm_eps=self.model_args.qk_norm_eps,
                        qk_norm_type=self.model_args.qk_norm_type,
                        use_cache=use_cache,
                        layer_idx=layer_idx,
                        kv_cache=self.kv_cache,
                        is_causal=self.model_args.is_causal,
                        padding_mask=padding_mask
                    )

            # Apply final RMSNorm
            x = self.rms_norm(x) # [B, H*W, d_model]

            # Apply patch prediction head
            logits = self.prediction_head(x) # [B, H*W, codebook_size]
            
            return logits

def test_model(use_cache:bool=False, log_grads:bool=True):
    model_args = ModelArgs()
    model = SpatialTransformer(model_args)
    B, H, W = 20, 64, 64
    x = torch.randn(
        B, model_args.C_in, H, W, device=model_args.device, dtype=model_args.dtype
    )
    logits = model(x, use_cache)
    if log_grads:
        loss = logits.sum()
        loss.backward()
        for name, param in model.named_parameters():
            print(f"{name}: {param.grad}")
    return logits

if __name__ == "__main__":
    logits = test_model(True, False)
    print(logits.shape)
