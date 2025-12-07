## Implementations
- [ ] Implement VQVAE for spatial transformer
- [ ] Implement VQVAE for spatio-temporal transformer

## All Modules
- [ ] Fix autocast placement for all modules
- [ ] Research accelerated attention for all modules (Flash Attn. 3)
- [ ] Research accelerated MLP variant

## Models
- [ ] Add generation loop for spatio-temporal transformer
- [ ] Add generation loop for spatial transformer
- [ ] Add generation loop for text transformer
- [ ] Add logging to all KV caching modules

## Testing
- [ ] Add padding tests for encoder
- [ ] Add padding & causal tests for spatio-temporal transformer
- [ ] Add padding & causal tests for spatial transformer
- [ ] Add padding & causal tests for text transformer
- [ ] Add KV caching tests for spatial transformer

## Data
- [ ] Add data setup for text transformer
- [ ] Add data setup for spatio-temporal transformer
- [ ] Add data setup for spatial transformer
- [ ] Add data setup for encoder transformer

## Training
- [ ] Add training loop for text transformer
- [ ] Add loss & perplexity computation for text transformer
- [ ] Add training components for text transformer

## Evaluation
- [ ] Add validation step for each model
- [ ] Add metric computation (loss, perplexity, reconstruction quality)
- [ ] Add early stopping & best checkpoint saving
- [ ] Benchmark generation speed and memory usage

## GPUs
- [ ] Find GPUs (ideal: 2 5090s)
- [ ] Change default data type to bfloat16 when available
- [ ] Change default device to cuda when available
- [ ] Work on triton kernels for model bottlenecks

## Checkpoints
- [ ] Add checkpointing logic for text transformer
- [ ] Add checkpointing logic for spatio-temporal transformer
- [ ] Add checkpointing logic for spatial transformer
- [ ] Add checkpointing logic for encoder transformer

## Examples
- [ ] Write example usage for text transformer
- [ ] Write example usage for spatio-temporal transformer
- [ ] Write example usage for spatial transformer

## README.md
- [ ] Write table of contents for README
- [ ] Write advancements of this model
- [ ] Compare to previous models

## Utilities
- [ ] Add logger setup
- [ ] Add generation utilities

## Research
- [ ] Research VQVAE for spatial transformer
- [ ] Research VQVAE for spatio-temporal transformer
- [ ] Research action model
- [ ] Research latent model
- [ ] Research dynamics model

## Writing
- [ ] Start drafting technical report
