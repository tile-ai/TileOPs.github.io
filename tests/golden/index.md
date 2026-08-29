# Benchmarks

!!! info "Nightly snapshot"

    **GPU** NVIDIA H200 · **commit** [`0123456789ab`](https://github.com/tile-ai/TileOPs/commit/0123456789abcdef0123456789abcdef01234567) · **run date** 2026-01-01 · **5 ops**, 8 workloads
    · [nightly run](https://github.com/tile-ai/TileOPs/actions/runs/1234567890)

## Environment

| | |
| --- | --- |
| timer | `cupti` |

Not published by this run: `image`, `driver`, `cuda`, `torch`, `tilelang`.

## Method

- **One process, common inputs.** Every implementation of an op is timed on the same tensors in the same process, in forward and then reversed order so drift does not land on whichever ran last.
- **A fixed warmup and measurement budget** per implementation, reported as the median over however many samples fit in it, with L2 cleared between iterations.
- **Compilation and workspace setup excluded.**
- **Device time is what is compared** — the union of the intervals the device spent executing the call's kernels, collected through CUPTI. A run that cannot collect device activity fails rather than falling back to a different clock.

## Coverage

- **5 of 5 ops** are measured against a real alternative — a tuned library kernel or a native PyTorch op — on the identical workload. The rest run against an eager reference only, which is not a bar worth reporting a win against.
- **Absent from every table**: 1 workloads errored and 1 were skipped in this run.

[How these numbers are taken](reading.md)

## Data

| Page | Ops | Workloads |
| --- | --- | --- |
| [Linear Attention & SSM](linear-attention.md) | 2 | 4 |
| [Elementwise & Reduction](elementwise-reduction.md) | 3 | 4 |
