# Benchmarks

!!! info "Nightly snapshot"

    **GPU** NVIDIA H200 · **commit** [`b4fc786377c7`](https://github.com/tile-ai/TileOPs/commit/b4fc786377c75ae0871688b8f18d0274de2dccde) · **run date** 2026-09-05 · **181 ops**, 1047 workloads
    · [nightly run](https://github.com/tile-ai/TileOPs/actions/runs/33982693170)

    Page rendered 2026-09-06 02:39 UTC from the [latest snapshot](https://github.com/tile-ai/TileOPs-nightly/tree/snapshots).

## Environment

| | |
| --- | --- |
| image | `ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-afcebed1-2` |
| gpu | `NVIDIA H200` |
| driver | `595.71.05` |
| cuda | `13.2` |
| torch | `2.13.0+cu132` |
| tilelang | `0.1.11+cu132.gitafcebed1` |
| timer | `cupti` |
| memory_clock_mhz | `3201.0` |
| mig | `Disabled` |
| power_limit_w | `700.0` |
| sm_clock_max_mhz | `1980.0` |
| sm_clock_mhz | `1500.0` |

## Method

- **One process, common inputs.** Every implementation of an op is timed on the same tensors in the same process, in forward and then reversed order so drift does not land on whichever ran last.
- **A fixed warmup and measurement budget** per implementation, reported as the median over however many samples fit in it, with L2 cleared between iterations.
- **Compilation and workspace setup excluded.**
- **Device time is what is compared** — the union of the intervals the device spent executing the call's kernels, collected through CUPTI. A run that cannot collect device activity fails rather than falling back to a different clock.

## Coverage

- **178 of 181 ops** are measured against a real alternative — a tuned library kernel or a native PyTorch op — on the identical workload. The rest run against an eager reference only, which is not a bar worth reporting a win against.
- **Absent from every table**: 0 workloads errored and 13 were skipped in this run.

[How these numbers are taken](reading.md)

## Data

| Page | Ops | Workloads |
| --- | --- | --- |
| [Elementwise](elementwise.md) | 70 | 404 |
| [RoPE](rope.md) | 6 | 13 |
| [Reduction](reduction.md) | 21 | 90 |
| [Normalization](normalization.md) | 10 | 69 |
| [Conv & Pool](conv-pool.md) | 16 | 108 |
| [GEMM](gemm.md) | 7 | 67 |
| [Quantization](quantization.md) | 2 | 4 |
| [Attention](attention.md) | 12 | 63 |
| [MoE](moe.md) | 9 | 52 |
| [Linear Attention](linear-attention.md) | 17 | 131 |
| [SSM](ssm.md) | 7 | 35 |
| [Other](other.md) | 4 | 11 |
