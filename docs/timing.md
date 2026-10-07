# How a benchmark is timed

The nightly benchmark measures one row per workload per op and reports
`device_busy_ms`: the union of the execution intervals on the device of every kernel
that call produces. CUPTI records each kernel's device-side start and end, and an
external correlation id attributes the kernel to an iteration. L2 is cleared before
every iteration, warm-up runs for 25 ms and measurement for 100 ms, and the result is
the median.

**Every number in the tables is therefore time the device spent executing kernels. It
excludes the host cost of issuing the call and the gaps between kernels. Reading the
tables requires nothing more than this.**{ .keystone }

The remaining sections are for reference:

- [How one measurement runs](#how-it-runs): the pseudocode, and its five choices:
  calibration, the iteration count, clearing L2, attribution, and failing closed.
- [What is measured](#what-is-measured): the definition of `device_busy_ms`, and why
  the gaps between kernels are left out.
- [Why not wall-clock time](#why-not-wall-clock): at decode sizes, CUDA events cannot
  measure the execution time of a small kernel.
- [When to change how you measure](#when-to-change): needed only when writing a
  benchmark, including the cases this method cannot measure.

Every number below was measured on an H200, in the `tileops-runner:cu132-torch2.13`
image.

## How one measurement runs {#how-it-runs}

```python
from benchmarks.timing import bench_kernel

samples = bench_kernel(op, args=(x, weight))   # one Sample per iteration
```

A benchmark rarely calls `bench_kernel` directly. It goes through
`bench.Runner(op, case).compare()`, which takes medians over these samples and computes
the derived columns; [Writing benchmarks](user-guide/benchmark/writing.md) shows how.

Inside, `bench_kernel` has three stages: collect, attribute and measure.

```python
# Collect: each call runs under its own iteration number
with _phase_session():                          # kernels + copies + mappings + launch APIs
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

The five choices, and the reason for each:

1. **Calibrate.** Three calls estimate the cost of one call.
2. **Convert that into an iteration count.** The budgets, 25 ms of warm-up and 100 ms of
   measurement, are divided by the per-call cost, and the result is clamped to
   `[10, 200]`. A short op therefore gets more samples, and a long one does not have to
   run 200 times.
3. **Clear L2 before every iteration, and drain the device.** Without the clear, the
   first iteration reads from HBM and every later one from L2, so the median reports the
   best case of a full cache hit. Draining keeps the previous iteration from overlapping
   this one. An implementation that restores overwritten inputs does so in its `reset`,
   which runs before the clear, so the data it writes does not stay in L2.
4. **Collect and attribute.** Each iteration pushes its iteration number as CUPTI's
   external correlation id, so every launch issued inside it carries that id; the
   correlation id in a kernel record maps back to the iteration number.
   **Attribution does not use timestamps**: which iteration a kernel belongs to is
   written in its record, independent of when it ran. A kernel shorter than the host
   overhead is therefore attributed as reliably as a long one, and a call whose kernel
   count varies between iterations can still be measured.
5. **Fail closed.** Three attribution failures each raise a different error and produce
   no number:

| Case | Raises | Meaning |
| --- | --- | --- |
| CUPTI discarded records | `_CUPTIRecordsLostError` | the iteration did run, but the reading is lost; the whole phase is measured again, up to 3 attempts in all, with a 4× larger buffer each time |
| Nothing discarded, but a kernel carries no iteration number | `_OffThreadLaunchError` | the kernel was launched by a thread that never pushed an iteration number |
| Nothing discarded, and one iteration has no kernels at all | `_CUPTIAttributionError` | that call never ran on the device |

## What is measured {#what-is-measured}

**`device_busy_ms` is the length of the union of the execution intervals on the device
of every kernel one call produces.** A CUPTI kernel record gives the device-side start
and end of execution, excluding the host cost of issuing the call. There are three
cases:

- **A single-kernel call**: the kernel's execution time on the device.
- **A multi-kernel call**: the union of the intervals, that is, the total time during
  which at least one of the call's kernels was executing. Two concurrent kernels are not
  counted twice, because the sum of the two would be SM time, not the time the device was
  busy.
- **The gaps between kernels**: not counted.

A gap is left out because its cause cannot be determined. The device really was idle,
but the cause is either the op's own data dependency or the CPU not having issued the
next kernel yet, and CUPTI's records do not distinguish the two. A quantity whose cause
cannot be determined cannot be used to judge an implementation.

`tflops` and `bandwidth_tbs` divide by the same quantity. They describe the throughput
reached while the device was executing, and a denominator that included idle time
within the call would lower them systematically.

Defined this way, the number does not depend on how fast the host is. Changing CUPTI's
collection buffer from 256 KB to 32 MB raises the median `latency_ms` of one
three-kernel call from 35 us to 2068 us, while `device_busy_ms` stays at 19.1 us. A host
that issues kernels late does not change any kernel's execution time; it only spreads
the kernels apart on the timeline, and the length of the union is unchanged.

## Why not wall-clock time {#why-not-wall-clock}

At decode sizes, an op can finish faster than the Python call that launched it. Four
methods applied to one 3 us kernel give four readings:

| Method | Reading |
| --- | --- |
| CUPTI kernel records | 1.95 us |
| A pair of CUDA events per iteration | 6.03 us |
| One pair of events around the loop, divided by the iteration count | 6.07 us |
| CUDA graph replay | 4.30 us |

The device executed for 1.95 us. The 6 us the event methods read is the interval at
which the CPU issues the next call, not the kernel's execution time. **This is the only
reason TileOPs times with CUPTI**, and it is why a row that fell back to CUDA events
cannot be compared with the other rows: in that row `device_busy_ms` and `latency_ms`
hold the same number, and the `timing` field records `cuda-events`.

## Comparing several implementations {#comparing}

To compare implementations within one case, `bench.Runner.compare()` times each implementation
twice, in the order A B C C B A, and takes the median over the samples of both passes.

In a fixed order, the implementation that runs first and the one that runs last see
different clocks and temperatures, and that difference reads as a difference between
the implementations. A symmetric order puts each implementation's two passes in the
first and second half of the case, which cancels monotonic drift to first order. Two
details:

- **The budget is split, not doubled.** Each pass gets 12.5 ms of warm-up and 50 ms of
  measurement, with half the iteration bounds. The symmetric order is meant to cancel
  drift, not to add samples, so the sample count matches that of timing one
  implementation.
- **Both passes must use the same timing method.** When one pass uses CUPTI and the
  other falls back to CUDA events, `compare()` raises instead of pooling the results,
  since pooling would put two kinds of measurement into one median.

## When to change how you measure {#when-to-change}

The default case needs no change: one kernel per call, called through the Op
interface, timed by `bench_kernel`, with no other thread using the GPU. Most ops are in
this case today. Eight cases need separate handling:

| Your case | If you ignore it | What to do |
| --- | --- | --- |
| The timed closure contains `Tensor.backward` or `torch.autograd.grad` | the backward kernels come from the autograd engine's own thread, carry no iteration number, and the case raises instead of producing a figure | Drive a single fused node with `backward_of(out)`; for a chain, set `torch.autograd.set_multithreading_enabled(False)` |
| Another thread in the process uses the GPU, or the timed closure uses CUPTI's `CUSTOM0` external id | those kernels carry no iteration number, or the closure overwrites the one the timer set, and it raises either way | Have the timed call launch its own work; use `CUSTOM1` / `CUSTOM2` instead |
| The op produces its result through `copy_`, as in-place elementwise ops and MoE's write-back do | the timer collects the copy but by default leaves it out of `device_busy_ms` and reports it as `uncounted_copy_ms`, so the reading is too low | Set `count_copies=True` in the op's case entry under `benchmarks/_cases/`; every implementation's reading for that case then includes the copies |
| An implementation writes into one of its inputs in place, or keeps state between calls | later iterations start from different data, and `compare()` raises before timing because the shared `case.inputs` changed | Give that implementation a private copy of the argument and restore it in `reset`, as in [`bench.Implementation`](user-guide/benchmark/writing.md#implementation); `reset` runs before the L2 flush and stays out of the reading |
| One call launches several kernels | the gaps between kernels land in `latency_ms`, so a comparison by it against a fused implementation charges the gaps to the multi-kernel side | Draw conclusions from `device_busy_ms` only; `latency_ms` is comparable only between rows with equal `n_kernels` |
| One call takes more than 10 ms | the iteration count hits the floor of 10, the wall-clock time far exceeds the 100 ms budget, and p10/p90 over 10 samples are coarse | Accept the longer wall-clock time, or state an iteration count and the sample size |
| You want a kernel-level benchmark | the op has no spec, so shapes and roofline have to be written by hand and the spec validator cannot see them | Measure through the Op interface and write a [spec](user-guide/manifest/index.md) |
| You are adding an external baseline | moving the baseline's input conversion out of its timed region makes this repository carry that time instead | Keep the conversion inside the baseline's timed region; where that baseline is the reason the benchmark exists, require its dependency and let the import fail when it is missing |
