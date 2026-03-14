
# Reduction Operators

## Softmax

### `SoftmaxOp`

Numerically stable softmax.

```python
from tileops.ops import SoftmaxOp

op = SoftmaxOp(dtype=torch.float16)
output = op.forward(x, dim=-1)
```

### `LogSoftmaxOp`

Log-softmax for numerical stability in loss computation.

### `LogSumExpOp`

Log-sum-exp reduction.


## Arg Reductions

### `ArgmaxOp` / `ArgminOp`

Index of maximum/minimum value along a dimension.

