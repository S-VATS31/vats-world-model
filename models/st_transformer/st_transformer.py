import torch
import torch.nn as nn
from torch import Tensor
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint

from configs.st_transformer.model_args.xsmall import ModelArgs
from models.st_transformer.patch_embed import PatchEmbed3D
from models.st_transformer.temporal_cache import TemporalKVCache
from models.st_transformer.st_block import SpatioTemporalTransformerBlock

class SpatioTemporalTransformer(nn.Module):
    """Spatiotemporal transformer model.
    
    Args:
        model_args (ModelArgs): Model parameters.
    """
    def __init__(self, model_args: ModelArgs):
        super().__init__()

        self.model_args = ModelArgs

        # Set up patch embeddings
        self.patch_embed = PatchEmbed3D(
            C_in=model_args.C_in,
            d_model=model_args.d_model,
            patch_size=model_args.patch_size,
            use_proj_bias=model_args.use_conv3d_bias,
            device=model_args.device,
            dtype=model_args.dtype
        )
        
        # Set up KV cache
        self.kv_cache = TemporalKVCache(
            max_batch_size=model_args.max_batch_size,
            max_frames=model_args.max_frames,
            num_heads=model_args.num_heads,
            head_dim=model_args.d_model//model_args.num_heads,
            num_layers=model_args.num_layers,
            device=model_args.device,
            dtype=model_args.dtype
        )

        # Set up spatiotemporal layers
        self.layers = nn.ModuleList([
            SpatioTemporalTransformerBlock(
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

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for PyTorch modules.
        
        Args:
            module (nn.Module): PyTorch module to be initialized.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.RMSNorm):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)

    def forward(self, x: Tensor, use_cache: bool = False) -> Tensor:
        """Forward pass of SpatioTemporalTransformer.
        
        Args:
            x (Tensor): Input tensor of shape [B, C, T, H, W].
            use_cache (bool): Whether to use KV caching or not.
        
        Returns:
            Tensor: Logits tensor of shape [B, T, H*W, codebook_size].
        """
        with autocast(device_type=x.device.type):
            # x: [B, T, H*W, d_model]
            x, THW_patches, padding_mask = self.patch_embed(x)
            x = self.dropout(x)
            T_patch, H_patch, W_patch = THW_patches

            # Loop through transformer layers
            for layer_idx, layer in enumerate(self.layers):
                if self.model_args.gradient_checkpointing:
                    x = checkpoint(
                        layer,
                        x,
                        H_patch,
                        W_patch,
                        T_patch,
                        self.model_args.use_qk_norm,
                        self.model_args.use_mqa,
                        self.model_args.qk_norm_eps,
                        self.model_args.qk_norm_eps,
                        self.model_args.is_causal,
                        use_cache,
                        self.kv_cache,
                        layer_idx,
                        padding_mask,
                        self.model_args.attention_interleave,
                        use_reentrant=False
                    )
                else:
                    x  = layer(
                        x,
                        H_patch=H_patch,
                        W_patch=W_patch,
                        T_patch=T_patch,
                        use_qk_norm=self.model_args.use_qk_norm,
                        use_mqa=self.model_args.use_mqa,
                        qk_norm_eps=self.model_args.qk_norm_eps,
                        qk_norm_type=self.model_args.qk_norm_type,
                        is_causal=self.model_args.is_causal,
                        use_cache=use_cache,
                        kv_cache=self.kv_cache,
                        layer_idx=layer_idx,
                        padding_mask=padding_mask,
                        attention_interleave=self.model_args.attention_interleave
                    )
            
            # Apply final RMSNorm
            x = self.rms_norm(x)

            # Apply prediction head to get logits
            logits = self.prediction_head(x) # [B, T, H*W, d_model]

            return logits

def test_model(use_cache:bool=False):
    model_args = ModelArgs()
    model = SpatioTemporalTransformer(model_args)
    B, C, T, H, W = 40, 3, 65, 39, 37
    x = torch.randn(B, C, T, H, W, device="cpu", dtype=torch.float32)
    out = model(x, use_cache)
    return out

if __name__ == "__main__":
    out = test_model(True)
    print(out.shape)
