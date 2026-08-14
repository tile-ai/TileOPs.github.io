# TileOPs

Spec-driven LLM operators across backends — built by agents. The spec is the
source: kernels are derived from it and judged against it.

## Installation

```bash
pip install tileops
```

## Quick Start

An op commits to nothing at construction. Shapes and dtype come from the inputs
of the call, and the specialized kernel is built and cached on first use.

```python
import torch
from tileops.ops import GemmOp

a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

op = GemmOp()                        # NT by default: a=[M, K], b=[N, K]
d = op(a, b)                         # -> [M, N]
flops, nbytes = op.eval_roofline()   # what the call had to do and move
```

## Links

- [Benchmarks](benchmarks/index.md) — every op measured nightly against the
  alternatives, on an H200
- [GitHub](https://github.com/tile-ai/TileOPs)
- [Development guide](https://github.com/tile-ai/TileOPs/blob/main/docs/development.md)
