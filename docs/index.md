# TileOPs

TileOPs is an operator library for large-model inference, built on
[TileLang](https://github.com/tile-ai/tilelang), where one set of operator
interfaces can be implemented by different backends on different hardware.

What sets it apart from a hand-written library is how it is organised: every
operator is declared as a spec first, and an agent then derives the
implementation from that spec. The spec is the only input to generation and the
standard the result is judged by — correctness against the reference the spec
names, performance against the bound the roofline model gives, neither of them a
judgement call. An implementation can therefore be regenerated from its spec,
while the reverse does not hold.

To a caller it is simply a set of operators: shapes and dtype come from the call,
the specialized kernel is auto-tuned and cached on first use and works under CUDA
graphs afterwards, and each op declares whether it supports
`torch.compile(fullgraph=True)`.

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

## Where to go next

- [torch.compile integration](torch-compile.md) — what an op looks like inside a
  compiled graph, and the conventions a caller keeps to
- [Adding a Hardware Backend](backends.md) — take over the ops on a class of devices with
  your own kernels
- [API Reference](api/linear-algebra.md) — constructor parameters and call
  signatures per op family
- [Benchmarks](benchmarks/index.md) — measured nightly on an H200 against the
  alternatives

## Links

- [GitHub](https://github.com/tile-ai/TileOPs)
- [Development guide](https://github.com/tile-ai/TileOPs/blob/main/docs/development.md)
