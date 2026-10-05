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

The pages run in the order a reader meets the ops in a model: the pointwise
transforms and the positional rotation, then the axis reductions and the
normalizations built on them, the windowed kernels, the matmul and the
quantization around it, then attention, the expert routing after it and sampling,
then the sequence-model kernels. FFT, mHC and Engram follow, and Trace,
a tool rather than an op, comes last. The Benchmarks pages use the same order.

| Page | What it covers |
| --- | --- |
| [Elementwise](elementwise.md) | unary and binary maps, activations, dropout, and the in-place forms |
| [RoPE](rope.md) | rotary position embedding — NeoX and interleaved layouts, Llama 3.1, YaRN, LongRoPE |
| [Reduction](reduction.md) | sums, extrema, arg-reductions, cumulative scans, softmax |
| [Normalization](normalization.md) | RMSNorm, LayerNorm, GroupNorm, BatchNorm and the fused variants |
| [Pooling](pool.md) | average, max and adaptive pooling, with and without indices, plus the chunked sequence mean |
| [Convolution](convolution.md) | forward convolution over 1D, 2D and 3D inputs |
| [GEMM](linear-algebra.md) | dense matmul — plain, batched, and the fp8 variants |
| [Quantization & Dequantization](quantization.md) | INT8, FP8 and INT4 quantization, and INT8 dequantization |
| [Attention](attention.md) | forward and backward attention, including the paged and decode kernels |
| [MoE](moe.md) | the routed mixture-of-experts FFN and its separately callable stages |
| [Top-k & Sampling](sampling.md) | top-k selection, logits masks (top-k, top-p, min-p) and sampling, including chain speculative sampling |
| [Linear Attention](linear-attention.md) | DeltaNet, Gated DeltaNet, Kimi Delta Attention and Gated Linear Attention |
| [Mamba](mamba.md) | the SSD scan, its decode step, and the chunked forms |
| [FFT](fft.md) | the discrete transform |
| [mHC](mhc.md) | Manifold-Constrained Hyper-Connections — the pre/post pair around a layer |
| [Engram](engram.md) | the Engram GateConv pair and its decode step |
| [Trace](trace.md) | the in-kernel timeline tracer, a tool rather than an op |

Two things this reference does not carry:

- **What each op is allowed to receive.** The authoritative dtype domains, shape rules
  and measured workloads are in the op's spec; see [Writing a
  Spec](../user-guide/manifest/index.md).
- **How fast it is.** Device time against the fastest alternative on each workload is on
  the [Benchmarks](../benchmarks/index.md) pages.

These pages are generated from the docstrings in TileOPs, so an op whose docstring is
thin reads thin here. The fix belongs upstream, in
[`src/tileops/ops/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/ops).
