# Performance Guides

"Is this fast?" is several different questions, each answered somewhere else on
this site. Start from the one you are actually asking.

| Question | Where | What it gives you |
| --- | --- | --- |
| Is this op competitive on my shape? | [Benchmarks](../benchmarks/index.md) | Nightly device time per op per workload, against the fastest other implementation of the same op |
| How much work does one call cost? | [Roofline](../design/roofline.md) | `op.eval_roofline()` returns the FLOPs and bytes one call moves, from the manifest's `roofline` field — the ceiling a measurement is read against |
| Where does the time go inside a kernel? | [In-Kernel Timeline Trace](trace-timeline.md) | Per-CTA timelines from `clock64()` markers: gaps, stalls, producer/consumer overlap that per-kernel profilers do not surface |
| How do I measure this myself? | [Testing & Benchmarking](../design/testing.md) | Where correctness tests end and profiling begins, and which harness owns which |
| What is the tracer's API? | [Trace](../api/trace.md) | `trace.range`, `trace.group`, `trace.run`, and the rest of `tileops.trace` |
| How are these numbers taken? | [Benchmarks](../benchmarks/index.md) | "How these numbers are taken" in that section: the compared quantity, the ratio's direction, what each colour means, and which baselines can rate an op |

## Guides

- [In-Kernel Timeline Trace](trace-timeline.md) — annotate a kernel body with
  markers, build it traced or stripped from one source, and read the resulting
  timeline. Markers cost nothing when tracing is off, so they can stay in
  production code. Most useful for warp-specialized kernels, where overlapping
  the TMA producer with the WGMMA consumer is the whole point.
