# 性能指南

「这个快不快」其实是几个不同的问题，答案分散在站点的不同位置。先确认你问的是哪一个。

| 问题 | 去哪里 | 能得到什么 |
| --- | --- | --- |
| 这个 op 在我的形状上有竞争力吗 | [性能数据](../benchmarks/index.md) | 每晚测得的 device time，逐 op 逐 workload 与同一 op 最快的其他实现对比 |
| 一次调用的工作量是多少 | [Roofline](../design/roofline.md) | `op.eval_roofline()` 返回一次调用的计算量与访存量，取自 manifest 的 `roofline` 字段 —— 读测量结果时对照的上限 |
| kernel 内部时间花在哪 | [核内时间线追踪](trace-timeline.md) | 由 `clock64()` 标记得到的逐 CTA 时间线：空隙、停等、生产者与消费者的重叠，这些是 per-kernel profiler 看不到的 |
| 我自己怎么测 | [测试与性能测量](../design/testing.md) | 正确性测试到哪里为止、性能测量从哪里开始，各由哪套驱动代码负责 |
| tracer 的 API 是什么 | [Trace](../api/trace.md) | `trace.range`、`trace.group`、`trace.run` 以及 `tileops.trace` 的其余部分 |
| 这些数字是怎么测的 | [性能数据](../benchmarks/index.md) | 该栏的「How these numbers are taken」：比较的是哪个量、比值的方向、各个颜色的含义、哪些 baseline 才能给一个 op 定级 |

## 指南

- [核内时间线追踪](trace-timeline.md) —— 在 kernel 体上加标记，从同一份源码构建出带追踪与不带追踪两个版本，再读出时间线。关闭时标记不产生开销，因此可以留在生产代码里。对 warp-specialized kernel 最有用 —— 让 TMA 生产者与 WGMMA 消费者重叠正是这类 kernel 的全部要点。
