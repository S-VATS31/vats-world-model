from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor, LongTensor
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint

from models.text_transformer.kv_cache import KVCache
from models.text_transformer.block import CausalTransformerBlock
from configs.text_transformer.model_args.xsmall import ModelArgs

class CausalTransformer(nn.Module):
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        
        self.model_args = model_args

        # Set up embeddings
        self.token_embed = nn.Embedding(
            model_args.vocab_size, 
            model_args.d_model,
            device=model_args.device,
            dtype=model_args.dtype
        )

        # Set up transformer layers
        self.layers = nn.ModuleList([
            CausalTransformerBlock(
                d_model=model_args.d_model,
                num_heads=model_args.num_heads,
                query_groups=model_args.query_groups,
                use_qkv_bias=model_args.use_qkv_bias,
                use_o_bias=model_args.use_o_bias,
                use_qkv_proj=model_args.use_qkv_proj,
                rope_theta=model_args.rope_theta,
                softmax_scale=model_args.softmax_scale,
                rms_norm_eps=model_args.rms_norm_eps,
                dropout_prob=model_args.dropout_prob,
                d_ffn=model_args.d_ffn,
                use_mlp_bias=model_args.use_mlp_bias,
                device=model_args.device,
                dtype=model_args.dtype
            ) for _ in range(model_args.num_layers)
        ])

        # Set up KV cache
        self.kv_cache = KVCache(
            num_heads=model_args.num_heads,
            num_layers=model_args.num_layers,
            head_dim=model_args.d_model//model_args.num_heads,
            max_batch_size=model_args.max_batch_size,
            max_seq_len=model_args.max_seq_len,
            device=model_args.device,
            dtype=model_args.dtype
        )

        # Set up RMSNorm
        self.rms_norm = nn.RMSNorm(
            model_args.d_model,
            eps=model_args.rms_norm_eps,
            device=model_args.device,
            dtype=model_args.dtype
        )

        # Set up dropout
        self.dropout = nn.Dropout(p=model_args.dropout_prob)

        # Set up language modeling head
        self.lm_head = nn.Linear(
            model_args.d_model,
            model_args.vocab_size,
            bias=False,
            device=model_args.device,
            dtype=model_args.dtype
        )

        # Tie weights if being used
        if model_args.use_weight_tying:
            self.lm_head.weight = self.token_embed.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights.

        Args:
            module (nn.Module): Module to initialize.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.RMSNorm):
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.ones_(module.weight)

    def forward(
        self, 
        input_ids: LongTensor,
        use_cache: bool = False,
        padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass of transformer.
        
        Args:
            input_ids (LongTensor): Input LongTensor of shape [B, T].
            use_cache (bool): Whether to use KV caching or not.
            padding_mask (Optional[Tensor]): Padding tensor of shape [B, T].

        Returns:
            Tensor: Output logits of shape [B, T, V].
        """
        with autocast(device_type=input_ids.device.type):
            # Ensure input_ids are int64 tensors
            if input_ids.dtype != torch.int64:
                input_ids = input_ids.to(torch.int64)
            # Project to d_model
            x = self.dropout(self.token_embed(input_ids)) # [B, T, d_model]

            # loop through layers
            for layer_idx, layer in enumerate(self.layers):
                if self.model_args.gradient_checkpointing:
                    x = checkpoint(
                        layer,
                        x,
                        self.model_args.use_qk_norm,
                        self.model_args.use_mqa,
                        self.model_args.qk_norm_eps,
                        self.model_args.qk_norm_type,
                        self.kv_cache,
                        layer_idx,
                        use_cache,
                        self.model_args.is_causal,
                        padding_mask,
                        use_reentrant=False
                    )
                else:
                    x = layer(
                        x,
                        use_qk_norm=self.model_args.use_qk_norm,
                        use_mqa=self.model_args.use_mqa,
                        qk_norm_eps=self.model_args.qk_norm_eps,
                        qk_norm_type=self.model_args.qk_norm_type,
                        kv_cache=self.kv_cache,
                        layer_idx=layer_idx,
                        use_cache=use_cache,
                        is_causal=self.model_args.is_causal,
                        padding_mask=padding_mask
                    )

            # Apply final RMSNorm
            x = self.rms_norm(x)

            # Get logits through LM head projection
            logits = self.lm_head(x) # [B, T, V]

            return logits

def test_model(use_pad:bool=True, use_cache:bool=False):
    model_args = ModelArgs()
    model = CausalTransformer(model_args)
    B, T = 16, 8
    input_ids = torch.randint(
        0, model_args.vocab_size, (B, T), 
        device=model_args.device, dtype=torch.int64
    )
    if use_pad:
        padding_mask = torch.randint(
            0, 2, (B, T), 
            device=model_args.device, dtype=torch.int64
        )
    else:
        padding_mask = None
    out = model(input_ids, use_cache, padding_mask)
    return out

if __name__ == "__main__":
    out = test_model(True, True)
    print(out.shape)
