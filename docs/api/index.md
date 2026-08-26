# API Reference

Every op here is used the same way: construct it once, then call it. The constructor
takes what the kernel is compiled with — tile sizes, dimensions treated as constants,
dtypes — and the call takes the tensors. Both appear under each op as `__init__` and
`forward`, where `forward` is what runs when you write `op(...)`.

```python
import torch
from tileops.ops import GemmOp

op = GemmOp()                        # construct once, reuse
d = op(a, b)                         # the specialized kernel is built on first call
```

The pages are ordered along the stack a caller works through: the dense matmul first,
then attention, its position encodings and its linear replacements, then the
elementwise and reduction primitives, then the convolutional family and FFT.

| Page | What it covers |
| --- | --- |
| [GEMM](linear-algebra.md) | dense matmul — plain, batched, and the fp8 variants |
| [Attention](attention.md) | forward and backward attention, including the paged and decode kernels |
| [RoPE](rope.md) | rotary position embedding — NeoX and interleaved layouts, Llama 3.1, YaRN, LongRoPE |
| [Linear Attention](linear-attention.md) | the linear-attention family: DeltaNet, GLA, KDA, and their kin |
| [Mamba](mamba.md) | the SSD scan, its decode step, and the chunked forms |
| [MHC](mhc.md) | multi-head compression |
| [Normalization](normalization.md) | RMSNorm, LayerNorm, GroupNorm, BatchNorm and the fused variants |
| [Elementwise](elementwise.md) | unary and binary maps, activations, and the in-place forms |
| [Reduction](reduction.md) | sums, extrema, arg-reductions, cumulative scans, softmax |
| [Convolution](convolution.md) | forward convolution over 1D, 2D and 3D inputs |
| [Pooling](pool.md) | average, max and adaptive pooling, with and without indices |
| [Dropout](dropout.md) | dropout with deterministic replay |
| [FFT](fft.md) | the discrete transform |
| [Quantization](quantization.md) | fp8 quantization |
| [Top-k](topk.md) | top-k selection |
| [Trace](trace.md) | the in-kernel timeline tracer, a tool rather than an op |

Two things this reference does not carry:

- **What each op is allowed to receive.** The authoritative dtype domains, shape rules
  and measured workloads are in the op's spec; see [Writing a
  Spec](../manifest.md).
- **How fast it is.** Device time against the fastest alternative on each workload is on
  the [Benchmarks](../benchmarks/index.md) pages.

These pages are generated from the docstrings in TileOPs, so an op whose docstring is
thin reads thin here. The fix belongs upstream, in
[`src/tileops/ops/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/ops).
