
# Attention Operators

## Flash Attention

### `MultiHeadAttentionFwdOp`

Multi-head attention forward pass using Flash Attention algorithm.

- **Supported architectures**: Ampere (SM 80/86), Hopper (SM 90)
- **Supported dtypes**: `float16`, `bfloat16`

```python
from tileops.ops import MultiHeadAttentionFwdOp

op = MultiHeadAttentionFwdOp(dtype=torch.float16)
output = op.forward(Q, K, V)
```

### `MultiHeadAttentionBwdOp`

Multi-head attention backward pass.

### `MultiHeadAttentionDecodeWithKVCacheOp`

Optimized decode-phase attention with KV-cache support.

### `MultiHeadAttentionDecodePagedWithKVCacheOp`

Decode attention with paged KV-cache for memory-efficient serving.


## DeepSeek Attention Variants

### `MultiHeadLatentAttentionDecodeWithKVCacheOp`

DeepSeek MLA (Multi-head Latent Attention) decode.

### `DeepSeekSparseAttentionDecodeWithKVCacheOp`

DeepSeek DSA (Dynamic Sparse Attention) decode.


## Linear Attention

### Engram

- `EngramLinearAttentionFwdOp`
- `EngramLinearAttentionBwdOp`
- `EngramLinearAttentionDecodeOp`

### Gated DeltaNet

- `GatedDeltaNetFwdOp`
- `GatedDeltaNetBwdOp`
- `GatedDeltaNetDecodeOp`
