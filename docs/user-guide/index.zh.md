# 使用指南

| 文档 | 内容 |
| --- | --- |
| [读写 manifest](manifest/index.md) | 系统如何使用 spec、描述 spec 所用的概念，以及如何写一份 spec |
| [op 如何选择 kernel](dispatch/index.md) | op 如何选中 kernel、如何新增 kernel，以及 backend 如何接入 |
| [添加新 op](../new-op.md) | 从一份 spec 到 `status: implemented` 的六步 |
| [接入 torch.compile](../torch-compile.md) | op 在编译图中的形态，以及调用时的约定 |
| [benchmark 的计时方法](../timing.md) | Benchmarks 页上的数字如何测得 |
| [接入新硬件 backend](../backends.md) | 用自己的 kernel 接管某一类设备上的 op |
