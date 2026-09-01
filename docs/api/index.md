# API Reference

Every op here is used the same way: construct it once, then call it. The constructor
takes what the kernel is compiled with — tile sizes, dimensions treated as constants,
dtypes — and the call takes the tensors. Both appear under each op as `__init__` and
`forward`, where `forward` is what runs when you write `op(...)`.

```python
import torch
from tileops.gemm import GemmFwdOp

op = GemmFwdOp()                        # construct once, reuse
d = op(a, b)                         # the specialized kernel is built on first call
```

The pages are ordered by how much an op composes: the pointwise transforms first, then
the axis reductions and the normalizations built on them, then the matmul and the
expert routing over it, then the windowed and spectral transforms, then the
sequence-model kernels built on all of the above. It is the order `tileops` declares
its op families in.

| Page | What it covers |
| --- | --- |
| [Elementwise](elementwise.md) | unary and binary maps, activations, dropout, and the in-place forms |
| [Reduction](reduction.md) | sums, extrema, arg-reductions, cumulative scans, softmax |
| [Normalization](normalization.md) | RMSNorm, LayerNorm, GroupNorm, BatchNorm and the fused variants |
| [Quantization](quantization.md) | fp8 quantization |
| [Top-k](topk.md) | top-k selection |
| [GEMM](linear-algebra.md) | dense matmul — plain, batched, and the fp8 variants |
| [Pooling](pool.md) | average, max and adaptive pooling, with and without indices, plus the chunked sequence mean |
| [Convolution](convolution.md) | forward convolution over 1D, 2D and 3D inputs |
| [FFT](fft.md) | the discrete transform |
| [MoE](moe.md) | the routed mixture-of-experts FFN and its separately callable stages |
| [RoPE](rope.md) | rotary position embedding — NeoX and interleaved layouts, Llama 3.1, YaRN, LongRoPE |
| [Attention](attention.md) | forward and backward attention, including the paged and decode kernels |
| [Linear Attention](linear-attention.md) | DeltaNet, Gated DeltaNet and gated linear attention |
| [Mamba](mamba.md) | the SSD scan, its decode step, and the chunked forms |
| [mHC](mhc.md) | Manifold-Constrained Hyper-Connections — the pre/post pair around a layer |
| [Engram](engram.md) | the Engram GateConv pair and its decode step |
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
