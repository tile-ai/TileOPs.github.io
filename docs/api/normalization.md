
# Normalization Operators

## Layer Normalization

### `LayerNormOp`

Standard layer normalization.

```python
from tileops.ops import LayerNormOp

op = LayerNormOp(dtype=torch.float16)
output = op.forward(x, weight, bias)
```

### `RmsNormOp`

Root mean square normalization (used in LLaMA, etc.).


## Fused Normalization

### `FusedAddLayerNormOp`

Fused residual add + layer normalization (avoids extra memory read/write).

### `FusedAddRmsNormOp`

Fused residual add + RMS normalization.

