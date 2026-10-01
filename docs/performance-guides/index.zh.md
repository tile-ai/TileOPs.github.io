# 性能指南

## Nightly 性能数据

1. [性能数据](../benchmarks/index.md) —— 每晚在 H200 上运行的性能测量，逐个 op、逐个 workload 给出 device time，并与同一 op 最快的其他实现对比。数字的取法与比值的读法见该栏的「How these numbers are taken」。

## 调优工具

1. [核内时间线追踪](trace-timeline.md) —— 在 kernel 体内加标记，运行后得到逐 CTA 的时间线，用于定位空隙、停等以及生产者与消费者的重叠情况。这部分信息是 per-kernel profiler 看不到的。追踪的 API 见 [Trace](../api/trace.md)。

## TileLang 性能调优最佳实践

1. [优化访存受限的 kernel](memory-bound/index.md) —— 说明访存受限在 roofline 上的位置，并给出 access pattern 影响带宽的两处（global memory 与 shared memory）各自的实测结论。

一次调用的计算量与访存量由 `op.eval_roofline()` 给出，取自 manifest 的 `roofline` 字段，是阅读测量结果时对照的上限。模型与字段规范见 [Roofline](../design/roofline.md)。
