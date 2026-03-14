
# Linear Algebra Operators

## GEMM

### `GemmOp`

General matrix multiplication with support for transpose modes.

- **Supported architectures**: Ampere (SM 80/86), Hopper (SM 90)
- **Supported dtypes**: `float16`, `bfloat16`, `float32`

```python
from tileops.ops import GemmOp

op = GemmOp(dtype=torch.float16)
C = op.forward(A, B)  # C = A @ B
```

Transpose modes: `NN`, `NT`, `TN`, `TT`

Also handles GEMV (matrix-vector) cases internally.

