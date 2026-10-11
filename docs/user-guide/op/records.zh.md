# 调用记录与调优

本页说明 `Op` 基类在调用之后提供的信息与操作：

- 最近一次调用的记录；
- 从记录求出的 roofline；
- 已构造的 kernel 的枚举；
- 调优。

## 1. last_call {#last-call}

`op.last_call` 返回最近一次完成调用的记录 `SignatureCall`：

- 还没有调用完成时，它抛出 `RuntimeError`；
- 失败的调用不替换记录，见[构造与调用 § 失败与撤销](lifecycle.md#failure)。

**表 1** `SignatureCall` 的字段

| No. | 字段 | 含义 |
| --- | --- | --- |
| 1 | `indices` | 签名中各 index 求出的值：`forall` 的标量与 dtype index、位于 index 位置的构造参数与 `let`；`forall` 中 `Seq[Int]` 类型的 index 不在其中 |
| 2 | `branch_key` | 签名检查采用的判别分支 |
| 3 | `device` | 签名检查确定的调用设备，基类据此选择 target |
| 4 | `tensors` | • 按签名中的名字，记录这次调用出现的输入、构造期张量与输出的 shape 与 dtype<br>• 调用方传入的 `out` 不单列 |
| 5 | `traffic` | 每个张量的读写次数，由 manifest 的 effect 规则给出 |
| 6 | `written`、`has_out` | 这次调用写入的输入，以及调用方是否传入了 `out` |
| 7 | `metadata` | 声明了 `values` 的 metadata 输入张量 |
| 8 | `stages` | 组合 op 每个 stage 中子 op 完成的调用，按完成顺序排列，见[组合 op](composite.md#stages) |

使用记录时注意：

- **记录对外按不可变对象使用。** `stages` 在记录成为 `last_call` 之前写入一次。
- **`metadata` 保存的是张量引用，不是拷贝。**
    - `metadata_values(name)` 在被调用时才读取张量内容，所以调用过程中不需要同步到 host。
    - 调用方之后修改这些张量，早先记录读到的值也会随之改变。

## 2. roofline {#roofline}

- **`eval_roofline()`**
    - 由 manifest 条目的 `roofline` 字段生成，对 `last_call` 求值，返回 `(flops, bytes)`。
    - 公式的语法与计时口径见设计文档 [Roofline](../../design/roofline.md)。
- **`roof_key()`**
    - 给出为这些 FLOPs 定价的计算单元，基类默认值为 `"cuda_core.fp32"`。
    - FLOPs 来自矩阵乘的子类覆盖它，通常返回从 `self.last_call` 读取的输入 dtype 对应的 `tensor_core_roof`，见设计文档 [Roofline § 1.4](../../design/roofline.md#14-compute-roof)。

子类还可以覆盖两个方法：

1. **`eval_roofline_read_bytes()`** 返回 `bytes` 中读的部分，供 NCU 的字节审计使用。
    - 基类用 `bytes` 减去签名确定的写入量。
    - 签名无法确定读的部分时，子类覆盖并返回 `None`。
1. **`roofline_data_terms()`** 返回由输入的值决定 FLOPs 或 bytes 的那些量。
    - 例如 DSA 这次调用选中的 key 数，与访问到的不同 `kv` 行数。
    - benchmark 把它记录在读数旁边，不参与 `(flops, bytes)`。
    - 基类返回空字典。

## 3. 枚举 {#enumerate}

**表 2** 枚举方法

| No. | 方法 | 返回 |
| --- | --- | --- |
| 1 | `iter_kernels()` | • 各接口的 entry 和持有的子 op 中的每个 `Kernel`，每个只返回一次<br>• 不遍历 target 构造的 kernel，它们由 `built_entries(interface)` 查看 |
| 2 | `built_entries(interface)` | • in-tree 路径：返回这个 kernel 接口已构造的 entry，按 `(实现类, identity)` 区分<br>• target 服务时：返回这个 op 的全部 target kernel，按设备与每个输入的 dtype、shape 区分，`interface` 不参与筛选 |
| 3 | `held_delegates()` | 组合 op 持有的子 op，按 stage 顺序 |

三个方法的用途：

- `built_entries()` 只用于测试与 benchmark 报告。
- `iter_kernels()` 还供 `request_tune()` 与 `kernel_config()` 使用。
- `held_delegates()` 还供失败时递归撤销子 op 使用。

调用路径不用这三个方法取得 kernel：

- in-tree 本体通过 `kernel_for` 取得；target 路径由基类按缓存键查找。
- 二者在缓存未命中时选择并构造 kernel，不因未命中而报错。
- 调用或实现不符合要求时仍会报错，见[构造与调用 § kernel 缓存](lifecycle.md#kernel)。

`iter_kernels()` 只枚举声明过的位置，不反射扫描属性：

- 枚举的位置：各接口的 entry 和持有的子 op。
- 反射扫描会漏掉嵌套得更深、或存放在无法识别类型的属性中的 kernel，且不报错。
- 只枚举声明的位置，遗漏就表现为缺少声明。

## 4. 调优 {#tune}

- **`request_tune()`：** 进入调优模式的唯一方式，op 与 kernel 都不接收 `tune` 构造参数。它把 op 及其持有的子 op 置为调优模式。
    - 对已构造的每个 kernel 调用 `Kernel.request_tune()`：程序已构建的 kernel 立即调优，尚未构建的在构建程序的那次启动时调优；没有 `autotune_configs` 的 kernel 不调优。
    - 之后构造的 entry，基类构造后立即对其中每个 `Kernel` 调用 `request_tune()`。
- **target：** 调优请求到不了 target 构造的 kernel，此时发出警告，每个实例只警告一次。
- **子类覆盖：** 以上是基类的行为，子类可以覆盖。`GemmW4A16FwdOp` 不支持通用调优：覆盖 `request_tune()`，发出警告，改用自己校准过的选择。
- **`kernel_config()`：** 返回 op 自己的配置；没有时，返回 `iter_kernels()` 中第一个有配置的 kernel 的配置。
