# 性能指南

## Nightly 性能数据

1. [性能数据](../benchmarks/index.md) —— 每晚在 H200 上运行的性能测量，逐算子逐 workload 给出 device time，并与同一算子最快的其他实现对比。数字怎么取、比值怎么读，见该栏的「How these numbers are taken」。

## 调优工具

1. [核内时间线追踪](trace-timeline.md) —— 在 kernel 体内加标记，运行后得到逐 CTA 的时间线，用于定位空隙、停等以及生产者与消费者的重叠情况。这是 per-kernel profiler 看不到的部分。追踪的 API 见 [Trace](../api/trace.md)。

## TileLang 性能调优最佳实践

1. [优化访存受限的 kernel](memory-bound/index.md) —— 访存受限在 roofline 上是什么位置，以及 access pattern 决定带宽的两处地方（global memory 与 shared memory）各自的实测结论。

一次调用的计算量与访存量由 `op.eval_roofline()` 给出，取自 manifest 的 `roofline` 字段，是读测量结果时对照的上限；模型与字段规范见 [Roofline](../design/roofline.md)。
