from typing import Optional

import torch.nn as nn
from torch import Tensor, LongTensor
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint

from models.text_encoder.block import TransformerBlock
from configs.text_encoder.model_args.xsmall import ModelArgs

class TransformerEncoder(nn.Module):
    """Complete encoder model.
    
    Args:
        model_args (ModelArgs): Model parameters.
    """
    def __init__(self, model_args: ModelArgs):
        super().__init__()

        self.model_args = model_args

        # Embedding layer
        self.embed = nn.Embedding(
            model_args.vocab_size, 
            model_args.d_model,
            device=model_args.device,
            dtype=model_args.dtype
        )

        # Set up layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                num_heads=model_args.num_heads,
                d_model=model_args.d_model,
                query_groups=model_args.query_groups,
                rope_theta=model_args.rope_theta,
                softmax_scale=model_args.softmax_scale,
                use_proj_bias=model_args.use_proj_bias,
                use_qkv_proj=model_args.use_qkv_proj,
                rms_norm_eps=model_args.rms_norm_eps,
                dropout_prob=model_args.dropout_prob,
                d_ffn=model_args.d_ffn,
                use_mlp_bias=model_args.use_mlp_bias,
                device=model_args.device,
                dtype=model_args.dtype
            ) for _ in range(model_args.num_layers)
        ])

        # RMSNorm and dropout
        self.rms_norm = nn.RMSNorm(
            model_args.d_model,
            eps=model_args.rms_norm_eps,
            device=model_args.device,
            dtype=model_args.dtype
        )
        self.dropout = nn.Dropout(p=model_args.dropout_prob)

        self.apply(self._init_weights)
    
    def _init_weights(self, module) -> None:
        """Weight initialization for Transformer encoder components.
        
        Args:
            module: Module to be initialized.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.model_args.d_model ** -0.5)

        elif isinstance(module, nn.RMSNorm):
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)

    def forward(
        self, 
        input_ids: LongTensor,
        padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass of complete transformer.
        
        Args:
            input_ids (LongTensor): Input tensor of shape [B, T].
            padding_mask (Optional[Tensor]): Padding tensor of shape [B, T].

        Returns:
            Tensor: Output tensor of shape [B, T, d_model].
        """
        with autocast(device_type=input_ids.device.type):
            x = self.dropout(self.embed(input_ids)) # [B, T, d_model]

            # Stack layers
            for layer in self.layers:
                if self.model_args.gradient_checkpointing:
                    x = checkpoint(
                        layer,
                        x,
                        self.model_args.use_qk_norm,
                        self.model_args.use_mqa,
                        self.model_args.qk_norm_eps,
                        self.model_args.qk_norm_type,
                        padding_mask,
                        use_reentrant=False
                    )
                else:
                    x = layer(
                        x,
                        self.model_args.use_qk_norm,
                        self.model_args.use_mqa,
                        self.model_args.qk_norm_eps,
                        self.model_args.qk_norm_type,
                        padding_mask,
                    )

            # Apply final RMSNorm
            x = self.rms_norm(x)

            return x
