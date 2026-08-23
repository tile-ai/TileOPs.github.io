# Performance Guides

Performance material comes in two kinds: numbers already measured, and tools for
locating a problem yourself.

- [Benchmarks](../benchmarks/index.md) — a nightly run on an H200, reporting
  device time per op per workload against the fastest other implementation of the
  same op. How the numbers are taken and how to read the ratio is set out in
  "How these numbers are taken" in that section.
- [In-Kernel Timeline Trace](trace-timeline.md) — annotate a kernel body with
  markers and read back a per-CTA timeline: gaps, stalls, and how far the
  producer and consumer overlap. None of that is visible to a per-kernel
  profiler. The API is [Trace](../api/trace.md).

The work one call costs comes from `op.eval_roofline()`, derived from the
manifest's `roofline` field — the ceiling a measurement is read against. The
model and the field specification are in [Roofline](../design/roofline.md).
