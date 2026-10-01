# TileOPs

TileOPs is an exploratory operator library for large-model inference, built on
[TileLang](https://github.com/tile-ai/tilelang). TileOPs is designed for agents,
and agents build the whole project. Such a project has to meet concrete
code-quality requirements: its structure stays consistent, it does not diverge or
bloat as ops are added, and its code stays maintainable. The design of TileOPs
serves three goals:

- **Maintainable.** Each op is declared by a spec, and an agent generates the
  implementation from it. The ops in a family share one set of interfaces and
  rules, and every new op and kernel follows them.
- **Verifiable.** The spec names the reference implementation correctness is
  judged against. Tests compare a kernel's output with that reference, and
  performance measurements compare its measured speed with the bound the roofline
  model gives. The acceptance criteria are fixed in advance, and the checks run
  automatically.
- **Tunable.** The roofline model gives the gap between each kernel and its
  performance bound. The [nightly benchmarks](benchmarks/index.md) compare each
  kernel with the fastest other implementation on the same hardware.

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

- [Blog](blog/index.md): technical explorations from building TileOPs.
- [User Guide](user-guide/index.md): reading and writing the manifest, bringing an op
  into `torch.compile`, how a benchmark is timed, and adding a hardware backend.
- [API Reference](api/index.md): the constructor parameters and call signatures of
  each op family.
- [Benchmarks](benchmarks/index.md): measured nightly on an H200, each workload
  against other implementations.

## Links

- [GitHub](https://github.com/tile-ai/TileOPs)
- [Development guide](https://github.com/tile-ai/TileOPs/blob/main/docs/development.md)
