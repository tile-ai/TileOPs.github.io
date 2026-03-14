
# Elementwise Operators

## Unary Operations (30+)

Math functions applied element-by-element:

| Op | Description |
|:---|:------------|
| `AbsOp` | Absolute value |
| `NegOp` | Negation |
| `ExpOp` / `Exp2Op` / `Expm1Op` | Exponential variants |
| `LogOp` / `Log2Op` / `Log10Op` / `Log1pOp` | Logarithm variants |
| `SqrtOp` / `RsqrtOp` | Square root / reciprocal square root |
| `SinOp` / `CosOp` / `TanOp` | Trigonometric |
| `SinhOp` / `CoshOp` / `TanhOp` | Hyperbolic |
| `AsinOp` / `AcosOp` / `AtanOp` | Inverse trigonometric |
| `SigmoidOp` | Sigmoid activation |
| `ReluOp` / `GeluOp` / `SiluOp` | Activation functions |
| `ClampOp` | Value clamping |
| `CeilOp` / `FloorOp` / `RoundOp` / `TruncOp` | Rounding |
| `BitwiseNotOp` | Bitwise NOT |
| `ReciprocalOp` | 1/x |
| `ErfOp` / `ErfcOp` / `ErfinvOp` | Error function variants |
| `SignOp` / `SgnOp` | Sign functions |

```python
from tileops.ops import GeluOp

op = GeluOp(dtype=torch.float16)
output = op.forward(x)
```


## Fused Operations

### `FusedGatedOp`

Fused gated activation: combines a binary operation with an elementwise activation in a single kernel launch (e.g., SwiGLU = `x * silu(gate)`).
