# TileOPs

TileOPs is an operator library for large-model inference, built on
[TileLang](https://github.com/tile-ai/tilelang). One set of op interfaces can be
implemented by different backends on different hardware.

TileOPs differs from a hand-written operator library in how it is organised: every
op is first declared as a spec, and an agent then generates the implementation from
that spec. The spec is the only input to code generation and the standard the result
is accepted against:

- correctness is judged against the reference implementation the spec names;
- performance is judged against the bound the roofline model gives.

Neither check depends on human judgement. An implementation can therefore be
regenerated from its spec at any time, while a spec cannot be derived from an
implementation.

To a caller, TileOPs is a set of ops that can be called directly. Shapes and dtype
are fixed at call time; the specialized kernel is built and cached on first use and
can then be used with CUDA graphs. Each op declares whether it supports
`torch.compile(fullgraph=True)`.

## Installation

```bash
pip install tileops
```

## Quick Start

An op binds no shape at construction. Shapes and dtype come from the tensors passed
to the call, and the specialized kernel is compiled and cached on the first call.

```python
import torch
from tileops.gemm import GemmFwdOp

a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)

op = GemmFwdOp()                        # NT by default: a=[M, K], b=[N, K]
d = op(a, b)                         # -> [M, N]
flops, nbytes = op.eval_roofline()   # what the call had to do and move
```

## Where to go next

- [User Guide](user-guide/index.md): reading and writing the manifest, bringing an op
  into `torch.compile`, how a benchmark is timed, and adding a hardware backend.
- [API Reference](api/index.md): the constructor parameters and call signatures of
  each op family.
- [Benchmarks](benchmarks/index.md): measured nightly on an H200, each workload
  against other implementations.

## Links

- [GitHub](https://github.com/tile-ai/TileOPs)
- [Development guide](https://github.com/tile-ai/TileOPs/blob/main/docs/development.md)
