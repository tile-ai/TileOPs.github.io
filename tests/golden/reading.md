# How these numbers are taken

Every data page answers one question: **how does TileOPs compare to the fastest alternative implementation of the same op, on the same workload?** Each op gets one table, with one row per workload. Nothing is averaged across workloads: every number on the page belongs to a single shape and dtype.

## The colour is the verdict

| | Meaning |
| --- | --- |
| <span class="perf-behind">0.74×</span> | Slower than the alternative — below 0.95×. |
| <span class="perf-par">1.02×</span> | Level with it — 0.95–1.05×, inside measurement noise. |
| <span class="perf-ahead">1.42×</span> | Faster than it — 1.05× and above. |
| <span class="perf-unrated">18.06×</span> | No fast alternative to compare against — only a functional reference (a name ending in `-ref`). The number means little. |
| <span class="perf-none">—</span> | No alternative at all ran on this workload. |

A ratio is the alternative's device time divided by ours, so **above 1 means TileOPs is faster**.

## Columns

| Column | Meaning |
| --- | --- |
| **Workload** | `W1`, `W2`, … — the key above each table spells each one out: the benchmark's own id for it, the dtype it ran at, and every input tensor as `name: shape, dtype`. Tensors sharing a shape are named together, and each carries its own dtype, so a `mask` in `bool` says so where it is read. After the tensors come the dimensions the op is sized by rather than shaped by (`m`, `n`, `k` for a GEMM, `num_experts` for MoE routing), then dimmed, the parameters the call did not leave at the signature's default. A quantity the others already fix — `max_seqlen_q` is `max(q_lens)` — is not repeated. |
| **Ratio** | `alt / ours` — the fastest alternative's device time divided by ours, the one number the colour grades. |
| **Device time** | Milliseconds the device spent executing the call's kernels — the union of their intervals. Every comparison on these pages uses it. |
| **Alternatives** | One line per other implementation measured on this workload, fastest first, with its own device time in ms. A tuned library kernel (`fla`, `mamba`, `fa3`, `triton`, …), a native PyTorch op (`torch`), or a name ending in `-ref` — an eager composition of PyTorch ops, which is not a bar worth reporting a win against. Divide any of them by our device time to get the ratio against that one. |
| **Throughput** | TFLOP/s: required FLOPs / device time. The count is analytic — the op's `eval_roofline` formula evaluated on the workload's own shapes, not a hardware counter — so it counts the work the problem demands, not the instructions the kernel issued. Padding, recompute or a masked-out tile is therefore invisible here, and the figure is only comparable between implementations of the same op on the same workload. |
| **SOL** | Share of the algorithmic speed-of-light: the fastest time physics allows for the workload, divided by our device time. The `Ratio` column says whether someone is faster today; SOL says how much faster anyone could ever be. Details below. |
| **Bound** | The resource that sets the workload's floor: `mem` (HBM traffic), `comp` (compute throughput), or `lat` — the workload is too small for the model to judge, and its SOL number greys out with it. |

How that device time is measured — what it counts, what it leaves out, and where it refuses to produce a number — is in [Benchmark Timing](../timing.md).

Each op's heading carries its workload count and its test outcome (✅ passed · ❌ failed · ⏭️ all skipped · `·` no test matched).

## Speed of light

**SOL** is *algorithmic* speed-of-light efficiency: `max(bytes / bandwidth, FLOPs / compute roof) / device time`, priced against the machine's *calibrated* ceilings — the bandwidth and compute rates microbenchmarks actually reach on this GPU, not the spec sheet. 100% means no implementation of this algorithm on this hardware can be faster.

Three statements delimit what a reading means:

1. **Bytes are the algorithm's minimum traffic** — each input read once, each output written once — not the DRAM traffic the kernel generated. A kernel that moves data twice scores low; that is the point.
2. **FLOPs follow the TileOPs counting convention** (a transcendental counts as one), not per-instruction hardware cost; the metric does not certify a special-function-bound kernel as at its limit.
3. **The compute roof is the unit an optimal implementation would use** — declared per op, never inferred from the running kernel, so a kernel on the wrong unit is measured against the right ceiling.

| SOL | Bound | Meaning |
| --- | --- | --- |
| <span class="perf-ahead">92%</span> | mem | At the achievable ceiling (from the at-ceiling line the roofline spec sets). The ceiling is an envelope over access mixes, and a kernel's own mix caps below it, so the line leaves room for every mix. Optimizing further buys at most the remainder. |
| 63% | mem | Headroom remains. |
| <span class="perf-unrated">41%</span> | <span class="perf-unrated">lat</span> | The workload is too small for the model to judge — launch overhead dominates the measurement, not the roofline — so the number is shown but not graded. |
| <span class="perf-unrated">⚠ 108%</span> | mem | Above the calibrated ceiling: the formula or the calibration is wrong. Never read it as a fast kernel. |
| `·` | `·` | An input is missing: no roofline formula, a non-CUPTI timing, or no GPU profile for the device. |

The model, its thresholds and the formula-audit machinery are specified in TileOPs [`docs/design/roofline.md`](https://github.com/tile-ai/TileOPs/blob/main/docs/design/roofline.md); the page imports that implementation rather than re-deriving it.

## Where the shapes come from

The snapshot records what each workload measured, not what it ran on: the shapes are read from the TileOPs [spec manifest](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/manifest), joined to a row by the label and dtype the benchmark id is built from. A workload the manifest does not declare — a benchmark written by hand rather than driven by a spec — shows that id alone, with no shapes under it.

## Empty cells

`·` means an input to that metric was not recorded, never that the value is zero: the op reported no FLOP count for that workload, or no alternative ran on it.

