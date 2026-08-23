# Benchmark timing

The nightly benchmark measures one row per workload per op, reporting `device_busy_ms`:
the union of the execution intervals of every kernel one call produces. CUPTI records
each kernel's device-side start and end, an external correlation id attributes it to one
iteration, L2 is cleared before every iteration, and 25 ms of warm-up and 100 ms of
measurement produce the median.

**So every number in the table is time the device spent executing kernels: none of the
host cost of issuing the call, and none of the gaps between kernels. For reading the
tables, that is the whole story.**{ .keystone }

The sections below are there when you need them:

- [How one measurement runs](#how-one-measurement-runs) — the pseudocode, plus
  calibration, the iteration count, clearing L2, attribution and failing closed.
- [What is measured](#what-is-measured) — what `device_busy_ms` is, and why the gaps
  between kernels are left out.
- [Why not wall-clock time](#why-not-wall-clock-time) — at decode sizes CUDA events
  cannot measure a small kernel.
- [When to change how you measure](#when-to-change-how-you-measure) — only needed when
  writing a benchmark yourself, and it covers what this method cannot measure.

Every number below was measured on an H200, in the `tileops-runner:cu132-torch2.13`
image.

## How one measurement runs

```python
from benchmarks.timing import bench_kernel

samples = bench_kernel(op, args=(x, weight))   # one Sample per iteration
```

A benchmark rarely calls it directly, going through `ManifestBenchmark.profile()` or
`.compare()` instead, which take medians over these samples and compute the derived
columns.

Inside, `bench_kernel` is three stages — collect, attribute, measure:

```python
# Collect: each call runs under its own iteration number
with _phase_session():                          # kernels + mappings + launch APIs
    for i in range(n_repeat):
        with _labelled(_PREPARE_ID):            # the L2 flush and other preparation
            prepare_one(i)
        with _labelled(i):                      # push i ... pop
            run_one(i)                          # the call being timed
        torch.cuda.synchronize()                # drain it, so the next iteration is clean
    kernels, iteration_of = _flush()            # read the records once, after the loop
    dropped = _read_dropped() if _drop_counter_is_live() else None

# Attribute: the correlation id a kernel carries decides whose iteration it is
for kernel in kernels:
    i = iteration_of.get(kernel["correlation_id"])
    if i == _PREPARE_ID:
        continue                                # preparation is not the timed work
    if i is None or not 0 <= i < n_repeat:
        orphans.append(kernel)                  # no id was ever pushed for this
    else:
        claimed[i].append(kernel)

# Measure: one sample per iteration
for i in range(n_repeat):
    busy.append(union(claimed[i]))              # the measured quantity, see below
    latency.append(max(end) - min(start))
    n_kernels.append(len(claimed[i]))
```

Five choices in that code, each for a reason:

1. **Calibrate.** Three calls estimate what one call costs.
2. **Convert that into an iteration count.** The budgets — 25 ms of warm-up, 100 ms of
   measurement — divide by the per-call cost, clamped to `[10, 200]`. A short op gets
   many samples; a long one need not run 200 times.
3. **Clear L2 before every iteration, and drain the device.** Without the clear, the
   first iteration reads from HBM and every later one reads from L2, so the median
   reports the best case of a full cache hit. Draining keeps the previous iteration from
   overlapping this one.
4. **Collect and attribute.** The iteration number is pushed onto CUPTI's external
   correlation id stack, and the correlation id a kernel record carries maps back to it.
   **Attribution does not look at timestamps** — which iteration a kernel belongs to is
   written in its record, independent of when it ran, so a kernel shorter than the host
   overhead is attributed as reliably as a long one, and a call whose kernel count
   varies between iterations still measures.
5. **Fail closed.** Three attribution failures raise three different errors and produce
   no number:

| Case | Raises | Meaning |
| --- | --- | --- |
| CUPTI discarded records | `_CUPTIRecordsLostError` | the reading is gone though the iteration did run — the whole phase is measured again, up to 3 times, asking for a 4× larger buffer each time |
| Nothing discarded, but a kernel carries no iteration number | `_OffThreadLaunchError` | a thread that never pushed an id launched it |
| Nothing discarded, and one iteration has no kernels at all | `_CUPTIAttributionError` | that call never reached the device |

## What is measured

**`device_busy_ms`: the union of the execution intervals of every kernel one call
produces.** A CUPTI kernel record's start and end are the device-side execution bounds,
with none of the host cost of issuing the call. Three cases:

- **A single-kernel call** — the kernel's execution time on the device.
- **A multi-kernel call** — the union of the intervals, that is, the total time at least
  one of the call's kernels was executing. Two kernels running concurrently are not
  counted twice; that would be SM time, not the time the device was busy.
- **The gaps between kernels** — not counted.

A gap is left out because it cannot be attributed. The device really was idle, but the
cause is either the op's own data dependency or the CPU not having issued the next
kernel yet, and CUPTI's records do not distinguish the two. A quantity whose cause is
unknown cannot judge an implementation.

`tflops` and `bandwidth_tbs` divide by the same quantity: they describe the throughput
reached while the device was executing, and a denominator that included in-call idleness
would depress them systematically.

Defined this way, the number is immune to how fast the host is. Changing CUPTI's
collection buffer from 256 KB to 32 MB takes the median `latency_ms` of one three-kernel
call from 35 us to 2068 us, while `device_busy_ms` stays at 19.1 us — a late host does
not change any kernel's own execution time, it only pushes them apart on the timeline,
and the union of the intervals is the same.

## Why not wall-clock time

At decode sizes an op can finish faster than the Python call that launched it. Four
methods on the same 3 us kernel read four numbers:

| Method | Reading |
| --- | --- |
| CUPTI kernel records | 1.95 us |
| A pair of CUDA events per iteration | 6.03 us |
| One pair of events around the loop, divided by the iteration count | 6.07 us |
| CUDA graph replay | 4.30 us |

The device executed for 1.95 us; the 6 us the event methods read is the rate at which
the CPU issues the next call, not the kernel's execution. **That is the only reason
TileOPs times with CUPTI**, and it is why a row that fell back to CUDA events cannot be
compared with the others: there `device_busy_ms` and `latency_ms` carry the same number,
and the `timing` field records `cuda-events`.

## Comparing several implementations

Comparing implementations within one case, `compare()` times each of them twice in the
order A B C C B A and takes the median over both passes.

In a fixed order the implementation that ran first and the one that ran last sit at
different clocks and temperatures, and that difference reads as a difference between the
implementations; a symmetric order puts each implementation's two passes in the first
and second half of the case, cancelling monotonic drift to first order. Two details:

- **The budget is split, not doubled.** Each pass gets 12.5 ms of warm-up and 50 ms of
  measurement, with half the iteration bounds — the point is symmetry, not more samples,
  so the sample count matches timing one implementation.
- **Both passes must use the same timing method.** One pass on CUPTI and the other
  fallen back to CUDA events raises rather than pooling, which would put one median over
  two kinds of measurement.

## When to change how you measure

The default case needs none of this: one kernel per call, through the Op interface, timed
by `bench_kernel`, with no other thread using the GPU. That is where most ops are today.
Seven cases call for a stop:

| Your case | If you ignore it | What to do |
| --- | --- | --- |
| The timed closure contains `Tensor.backward` or `torch.autograd.grad` | the backward kernels come from the autograd engine's own thread, carry no iteration number, and the case raises instead of producing a figure | Drive a single fused node with `backward_of(out)`; for a chain, set `torch.autograd.set_multithreading_enabled(False)` |
| Another thread in the process uses the GPU, or the timed closure uses CUPTI's `CUSTOM0` external id | those kernels carry no iteration number, or the closure overwrites the one the timer set, and it raises either way | Have the timed call launch its own work; use `CUSTOM1` / `CUSTOM2` instead |
| The op produces its result through `copy_` — in-place elementwise, MoE's write-back | that copy goes through `cudaMemcpyAsync`, which a kernel-only collection never sees, so the reading is short and nothing raises | Collect `MEMCPY` / `MEMSET` as a check and confirm the window has none |
| One call launches several kernels | the gaps between kernels land in `latency_ms`, so comparing by it against a fused implementation charges them to your side | Conclude from `device_busy_ms` only; `latency_ms` compares between rows of equal `n_kernels` |
| One call takes more than 10 ms | the iteration count hits the floor of 10, the wall-clock far exceeds the 100 ms budget, and p10/p90 over 10 samples are coarse | Accept the longer wall-clock, or state an iteration count and the sample size |
| You want a kernel-level benchmark | the op has no spec, so shapes and roofline have to be written by hand and the spec validator cannot see them | Measure through the Op interface and write a [spec](manifest.md) |
| You are adding an external baseline | moving the baseline's input conversion out of its timed region has this repository carry that time instead | Keep the conversion inside the baseline's timed region, and let the import fail where that baseline is the point |
