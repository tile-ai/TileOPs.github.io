# 性能指南

性能相关的内容分两类：一类是已经测出来的数字，另一类是自己动手定位问题的工具。

- [性能数据](../benchmarks/index.md) —— 每晚在 H200 上运行的性能测量，逐算子逐 workload 给出 device time，并与同一算子最快的其他实现对比。数字怎么取、比值怎么读，见该栏的「How these numbers are taken」。
- [核内时间线追踪](trace-timeline.md) —— 在 kernel 体内加标记，运行后得到逐 CTA 的时间线，用于定位空隙、停等以及生产者与消费者的重叠情况。这是 per-kernel profiler 看不到的部分。追踪的 API 见 [Trace](../api/trace.md)。

一次调用的计算量与访存量由 `op.eval_roofline()` 给出，取自 manifest 的 `roofline` 字段，是读测量结果时对照的上限；模型与字段规范见 [Roofline](../design/roofline.md)。
