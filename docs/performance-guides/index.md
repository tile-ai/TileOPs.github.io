# Performance Guides

## Nightly measurements

1. [Benchmarks](../benchmarks/index.md) — a nightly run on an H200, reporting
   device time per op per workload against the fastest other implementation of
   the same op. How the numbers are taken and how to read the ratio is set out
   in "How these numbers are taken" in that section.

## Tools for locating a problem

1. [In-Kernel Timeline Trace](trace-timeline.md) — annotate a kernel body with
   markers and read back a per-CTA timeline: gaps, stalls, and how far the
   producer and consumer overlap. None of that is visible to a per-kernel
   profiler. The API is [Trace](../api/trace.md).

## Tuning practice for TileLang

1. [Tuning Memory-Bound Kernels](memory-bound/index.md) — what memory-bound
   means on the roofline, and measured guidance on the two places an access
   pattern decides the bandwidth a kernel reaches: global memory and shared
   memory.

The work one call costs comes from `op.eval_roofline()`, derived from the
manifest's `roofline` field — the ceiling a measurement is read against. The
model and the field specification are in [Roofline](../design/roofline.md).
