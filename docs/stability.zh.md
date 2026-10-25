# API 稳定性

本页规定 TileOPs 支持哪些公开 API，以及一次发布可以改动什么。

## 公开模块

| 模块 | 包含的内容 |
| --- | --- |
| `tileops` | 各 family 模块、`Op`，以及它重新导出的 backend 名字。 |
| `tileops.<family>` | 一个 family 的 op，以及这些 op 的构造函数接收的值类型（例如 `tileops.moe.MaskedLayoutSpec`）。 |
| `tileops.backend` | target 选择，以及 in-tree 之外的 kernel 的登记。 |
| `tileops.authoring` | `Op`、`Kernel`、`KernelInterface`、`CallSpec`、`Entry`、`SignatureCall`：TileOPs 之外的代码编写新 op 所用的类。 |
| `tileops.manifest` | 它的 `__all__` 中的名字、`tileops.manifest.primitives` 模块，以及 spec 格式。 |
| `tileops.trace` | kernel 内的时间线追踪工具。 |

一个名字列在某个公开模块的 `__all__` 中，它就是公开的。
名字以下划线开头的模块、函数、类、属性或方法都是内部的，包括 `tileops._ops`、`tileops._kernels`、`tileops._utils` 与 `tileops._perf`。
内部名字在任何一次发布中都可能改动，不另行通知。

`tileops.elementwise.HardtanhFwdOp` 这类具体 op 公开的是构造与调用，不支持继承。

## 兼容性

在 0.x 版本内，minor 与 patch 发布对公开 API 只做增加：

- 公开的名字、签名、参数默认值、返回结构与文档写明的行为保持不变。
- 一次发布可以增加模块、名字、可选参数，或新接受的输入。
- 其他改动等到 major 发布。

`scripts/ci/check_api_compat.py` 在每个 PR 上对照最新发布做检查，一个调用若按最新发布写成、在 PR 上会被破坏，检查就失败。

## 支持范围

满足以下条件的调用受支持：

- 一个 op 实例同一时间只被一个线程、一个 CUDA stream 使用。
- 一个实例的 CUDA graph 回放不与该实例的 eager 调用重叠。
- 编译出的可调用对象（`torch.compile` 的输出）的生命周期不超过它调用的 op 实例。
- 调用不写入的输入是连续的，op 另有说明的除外。
- `Bwd` op 直接调用，不接入 `torch.autograd`。
- CUDA Graph 捕获支持 [`tests/cuda_graph_contract.py`](https://github.com/tile-ai/TileOPs/blob/main/tests/cuda_graph_contract.py) 列出的 op，限于这些 op 的 `cuda_graph` 测试所做的调用（[CUDA Graph validation](https://github.com/tile-ai/TileOPs/blob/main/docs/development.md#cuda-graph-validation)）。

## 缓存与日志

- 编译出的 kernel 与调优结果缓存在磁盘上的 `TILELANG_CACHE_DIR` 下，默认为 `~/.tilelang/cache`，跨进程重用。
- 调优进度经由 `logging` 模块输出，记在 `tilelang` logger 下，由应用决定去向；不向 stdout 打印，也不向工作目录写文件。
- 编译或运行失败的调优候选被跳过；设备内存耗尽或调用方代码出错会中止调优。

## 在 TileOPs 之外编写 op

TileOPs 之外的代码继承 `tileops.authoring.Op` 来编写新 op：

- 类以类属性 `spec` 给出它的 spec，格式与 manifest 条目相同（[manifest.md](design/manifest.md)）。
- `__init__` 先赋值 spec 中的参数，再调用 `super().__init__(target=target)`。
- 类在 `kernel_types` 中声明它的 `Kernel` 子类，在 `interfaces` 中声明它的 `KernelInterface` 调用位置，并编写 `forward`，用 `kernel_for(interface, call)` 取得 kernel。

之后像内置 op 一样以 `op(...)` 调用。
