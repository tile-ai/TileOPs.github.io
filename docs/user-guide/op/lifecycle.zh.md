# 构造与调用

本页说明一个 op 实例从构造到每次调用，在 `Op` 基类内部经过的步骤：

- `Op.__init__` 做什么；
- 调用从哪个入口进入基类；
- 一次调用的七步；
- 基类如何选择 target、安装实现表、缓存 kernel；
- 调用失败时撤销什么。

## 1. 构造：`Op.__init__` {#construct}

子类的构造函数按直接父类分两种写法：

- **直接继承 `Op`：** 先把 manifest 参数赋给同名属性，再调用 `super().__init__(target=target)`。
- **经过 family 基类：** 按 family 基类的签名传递参数，例如 RoPE 叶子类传 `input_layout`、`base`，归约类传 `dim`、`keepdim`；由 family 基类赋值这些参数，最终调用 `Op.__init__`。

两种写法都要在 `Op.__init__` 运行之前确定 manifest 参数的取值。

`Op.__init__` 保存 `target`，然后依次完成以下五步。每一步放在构造时各有原因；它不选择、也不构造任何 kernel。

**表 1** `Op.__init__` 的步骤

| No. | 步骤 | 为什么在构造时 |
| --- | --- | --- |
| 1 | 加载 backend 注册表 | • 必须在任何被 trace 的代码之前完成<br>• 实例的第一次调用可能已在 `torch.compile` 中，dynamo 无法 trace 模块导入与注册<br>• 构造按约定在编译之外进行 |
| 2 | 按 manifest 检查参数取值，把求出的值保存为 `_construction_indices` | • 参数在构造时确定，错误在构造时报出<br>• 求出的值每次调用都要用，只求一次 |
| 3 | 安装实现表并检查契约，见[第 4 节](#target)；同时建立两个 entry 缓存 `_entries_by_call` 与 `_built_entries`，均为空 | • 结果由 op 类的 `kernel_types` 与注册表决定，对一个实例不变<br>• 只解析类，不需要设备 |
| 4 | 登记实例，得到实例 key | • 编译边界的 custom op 按这个字符串 key 找回实例<br>• key 必须在 trace 之前存在，作为编译期常量，且不复用 |
| 5 | 建立绑定组的 `_target_kernels`，以及运行组的 `_delegates`、`_delegate_stages`、`_effect_branches`，全部为空，见[类结构与实例状态 § 实例状态](structure.md#state) | 基类读写这些字段时不判断字段是否存在 |

选择实现与构造 kernel 要等到调用时张量的 dtype、shape 与设备都已知：运行 in-tree kernel 时由 `kernel_for` 选择实现并取得 entry，见[第 5 节](#kernel)；由 target 服务时由基类调用 target 的 builder 构造 kernel，见[第 4 节](#target)。

**构造的顺序要求：**

- **manifest 参数的最终值要在调用 `super().__init__` 之前确定。** `Op.__init__` 检查的是调用它时 manifest 参数的取值。
    - 例如 `FusedMoEFwdOp` 与 `FusedMoESharedExpertFwdOp` 的 `activation` 可以由注入的 `experts` 决定。
    - 这两个 op 先按注入对象确定 `activation`，再调用 `super().__init__`。
- **不属于 manifest 参数的派生属性，可以在 `super().__init__` 之后赋值。** 例如池化 op 按维度展开的 kernel 大小。
- **组合 op 在 `super().__init__` 之后，通过 `delegate_for` 构造子 op。**

**构造参数：** 构造接收 manifest 参数、执行策略 `target`，以及 `injected_parameters` 列出的额外参数。调优不是构造参数，构造后用 `request_tune()` 请求。调用张量携带的以下两类事实不作为构造参数：

1. **张量的 shape。** 张量携带的维度随调用到达。构造时再接收一次，实例就可能与传入的张量不一致。
1. **输入张量的 dtype。**
    - 基类从本次调用的输入张量读取 dtype，用于签名检查；子类把它写进 call spec。
    - 选中哪个实现、是否复用已构造的 entry，由实现的选择规则与 `entry_for` 返回的 identity 决定，见[第 5 节](#kernel)。
    - 输出 dtype 由签名中的 dtype 表达式给出，见[调用与校验 § type inference](../manifest/calls.md#inference)。

例外与补充：

- **manifest 把一个维度或 dtype 声明为参数时，它是构造参数。**
    - 没有张量输入的 `AlibiFwdOp` 在构造时接收 `seq_len` 与 `out_dtype`。
    - `ProdFwdOp` 在构造时接收累加使用的 `dtype`，覆盖基类上默认为 `None` 的 `dtype`。
- **额外参数：** 类属性 `injected_parameters` 列出构造函数在 manifest 参数与执行策略之外允许的参数名，例如 `FusedMoEFwdOp` 的 `prepare_finalize` 与 `experts`。它只是 manifest validator 的白名单；参数由 family 自己定义和使用，不是基类的扩展机制。

**构造不探测设备：**

- 安装实现表只解析类，不读取设备属性。
- 调用期张量在构造之后才到达，可能位于进程还没有使用过的设备上。
- 不能运行 op 的设备，在第一次选择、构造或调用 kernel 时才被拒绝。
- manifest 声明的构造期张量（例如 `LongRoPEFwdOp` 的 `rescale_factors`）在构造时传入，构造检查读取它的 shape 与 dtype。

## 2. 调用入口 {#entry}

需要运行实现的 eager 调用都经过 `Op._run_call(inputs, body)`：

- `body` 是子类的计算本体 `forward`，每个 op 只写这一个方法；
- 进入 `_run_call` 的方式，取决于子类是否有编译边界。

**表 2** 调用入口

| No. | 入口 | 适用的 op | 进入 `_run_call` 的方式 | `body` |
| --- | --- | --- | --- | --- |
| 1 | `op(...)` | 有编译边界 | `__call__` 调用生成的 `_call_boundary`，经 custom op 进入 `_run_call` | `forward` |
| 2 | `op(...)` | 没有编译边界 | `__call__` 按 `forward` 的签名绑定参数，再调用 `_run_call` | `forward`，按参数名传入 |
| 3 | 以 meta 张量 eager 调用 `op(...)` | 有编译边界 | • 不进入<br>• 生成的 fake 运行签名检查并保存调用记录，按签名构造结果 | — |
| 4 | 被 trace 的 `op(...)` | 没有编译边界 | • 不进入，也不运行签名检查<br>• 实例已由 target 服务时调用 target 的 kernel，否则调用 `forward`<br>• 组合 op 的子 op 的 custom op 成为图中的节点：trace 时运行各自的 fake，执行图中的节点时才进入 `_run_call` | — |

表 2 的说明：

- **哪些 op 有编译边界：** 类定义时，基类为同时满足以下三条的 op 类生成编译边界，由 `_CompileBoundary` 注册 custom op，并把它们的名字写入类属性 `compile_op_names`：
    1. 有 manifest 条目；
    1. 条目有调用期张量输入；
    1. 条目没有 composition，即不是组合 op。
- **这是要求，不是选项：** 满足三条的 implemented op 都必须支持 `torch.compile(op, fullgraph=True)`，没有逐个 op 的豁免。validator 检查每个 implemented 条目：类的 `compile_op_names` 非空，当且仅当条目满足三条；测试另外要求每个这样的 op 都登记了 `fullgraph=True` 的冷编译测试，作为可编译的证据。
- **前两行的选择：** `__call__` 按 `compile_op_names` 是否为空，选择表 2 的前两行。
- **编译边界的细节**（custom op 如何生成、如何找回实例；被 trace 的路径受哪些限制：有编译边界时 trace 的是生成的 `_call_boundary`，没有编译边界的组合 op trace 的是 `forward`）见[接入 torch.compile](../../torch-compile.md)。
- **只有一份实现：** 运行实现的调用只有一个入口，所以检查、记录和失败处理只有一份实现。

## 3. 一次调用的七步 {#serve}

七步在时序中的位置见[索引页 § op 的生命周期](index.md#lifecycle)图 1。

**表 3** 一次调用的步骤

| No. | 步骤 | 内容 |
| --- | --- | --- |
| 1 | 开启调用 | • 压入当前线程的调用栈<br>• 收集本次调用期间子 op 完成的调用 |
| 2 | 检查签名 | 运行 `_SignaturePlan` 生成的检查，得到这次调用的记录 `SignatureCall` |
| 3 | 判断空写 | 至少有一个写入、且所有写入都没有元素时，不运行任何实现，按签名构造结果 |
| 4 | 选择 target | 实例尚未选定 target、且这次调用需要运行实现时选择，见[第 4 节](#target) |
| 5 | 执行 | 运行空写结果、target 的 kernel 或 `body`，三者取一 |
| 6 | 检查结果 | 按签名检查返回值 |
| 7 | 记录 | 1. 把子 op 的调用按 stage 写入记录的 `stages`<br>2. 弹出调用栈<br>3. 保存为 `last_call`<br>4. 报告给外层调用 |

签名检查与 dtype：

- 签名检查判断调用是否满足 manifest，dtype 在内。
- 调用期输入张量的 dtype 由这次调用决定，构造时无从比较；签名检查也会用到构造时已求出的 dtype index，例如构造期张量的 dtype。
- 通过检查的调用仍可能在选择 kernel 时被拒绝，例如没有实现支持这个 dtype，见 [op 如何选择 kernel § 一次调用如何找到 kernel](../dispatch/index.md#call-path)。

## 4. target 与实现表 {#target}

backend 接入 TileOPs 有两种方式，接入方法见 [backend 如何接入 TileOPs](../dispatch/backends.md#choose)。本节说明基类如何处理它们。

**表 4** backend 接入的两种方式

| No. | 方式 | 接管的范围 | 基类何时处理 |
| --- | --- | --- | --- |
| 1 | `target` | 整个 op | 调用的第 4 步选定，之后不再改变 |
| 2 | `register_kernel_type` | 为一个 kernel 接口新增一个实现，按它自己声明的适用范围与优先关系参与选择 | 构造时由 `Op.__init__` 并入实现表 |

### 选择 target {#select-target}

第一次需要运行实现的调用，按以下顺序选定 target：

1. 构造时传入的 `target`；
1. 为 `None` 时，取进程默认 target（`set_default_target` 设置）；
1. 仍为 `None` 时，按调用设备检测。

选定之后：

- **没有 backend 认领这个设备：** 运行 in-tree kernel，`serving_target` 为 `BUILTIN`。
- **有 backend 认领：** 运行这个 target 为 op 注册的 builder 构造的 kernel，子类的 `body` 不运行。
    - 基类调用 builder 时，按 `signature.inputs` 的顺序传入参数：出现的输入传 `TensorSpec`，未传入的可选输入传 `None`。
    - 实例上的 manifest 参数以关键字传入，见 [backend 如何接入 TileOPs § 替换整个 op](../dispatch/backends.md#target)。
- **选定的 target 没有为 op 注册 builder：**
    - 声明了自己 kernel 的 op 报 `OpNotAvailableError`，不退回 in-tree 实现；
    - 不声明 kernel 的组合 op 运行自己的组合。
- `serving_target` 返回选定的结果，选定之前为 `None`。

调用设备由签名检查确定，依次取：

1. 调用期张量所在的设备；
1. 声明的 `device` 参数；
1. 构造期张量所在的设备；
1. 三者都没有时，取当前 CUDA 设备。

CUDA 也不可用、且没有构造参数或进程默认 target 时，基类无从选择：

- 这次调用运行 in-tree 本体；
- 实例仍未选定 target，下一次调用再选择。

target 构造的 kernel 的缓存：

- 缓存键是调用设备，加上每个输入的 dtype 与 shape；
- 未传入的可选输入单独占位；
- 输出缓冲不进入缓存键。

target 被调用前，生成的检查已经运行，基类保证以下三点：

1. 除声明 `device: cpu` 的张量外，所有张量都在调用设备上；
1. 这次调用不写入的输入都是连续的，写入的输入在声明 `contiguous: true` 时是连续的；
1. 调用，包括调用方提供的输出缓冲，满足签名。

没有调用期张量输入的 op，按[调用与校验 § 调用设备](../manifest/calls.md#device)的规则决定设备。

### 安装实现表 {#install}

实现表 `_installed_kernel_types` 记录每个 key 对应的实现类。`Op.__init__` 依次：

1. 合并 `kernel_types` 与 backend 为这个 op 注册的 kernel type，key 不能重复；
1. 检查实现符合 kernel 接口的契约，不合要求时在构造时抛出 `TypeError` 或 `ValueError`：
    - 同时继承 `Kernel` 与所在 key 的 kernel 接口；
    - `entry_for` 是类方法；
    - `forward` 接受接口的参数；
    - 每个 key 属于某个接口；
    - 每个接口至多一个 `general` 实现；
    - `preferred_over` 只列同一接口的其他实现，`general` 实现不声明它；
    - `preferred_over` 不成环。

不声明自己 kernel 的组合 op，实现表为空；它的子 op 各自安装自己的实现表。

## 5. kernel 缓存 {#kernel}

本节区分三个词：

- **kernel type** 是 `Kernel` 的子类，即 [op 如何选择 kernel § 术语](../dispatch/index.md#terms)中的「实现」。它声明自己能在哪些设备上运行、适用于哪些调用，以及如何构造。
- **entry** 是 kernel type 的 `entry_for` 按一次调用构造出的对象，`kernel_for` 返回它，子类调用它完成计算。
- **kernel** 是可调用的对象：entry 中的 `Kernel` 实例，一个 entry 包含一个或多个；或 target 的 builder 返回的可调用对象。持有 TileLang 程序的 `Kernel` 实例，其程序可能在第一次启动时才构建。

一个 kernel type 按 identity 构造出多个 entry。以 `RMSNormOnChipKernel` 为例：

**表 5** kernel type 与 entry

| No. | | kernel type | entry |
| --- | --- | --- | --- |
| 1 | 例子 | `RMSNormOnChipKernel` 这个类 | 按 `n=4096`、fp16 等参数构造出的对象 |
| 2 | 何时存在 | 导入模块时 | 第一次需要它的调用到来时构造 |
| 3 | 携带什么 | 可用性与适用范围的声明（`supported_archs`、`refusal`、`preferred_over`）和 `entry_for` | 构造参数（由 identity 确定）与一个或多个 kernel |
| 4 | 数量 | 每个 key 一个 | 每个接口下，每个 `(kernel type, identity)` 至多一个，例如 `n=4096`、fp16 一个，`n=8192`、bf16 又一个 |
| 5 | 保存在 | `kernel_types`、`_installed_kernel_types` | `_built_entries` |
| 6 | 用途 | 基类据此选择这次调用的实现 | 子类调用它：`entry(x, weight)` |

`register_kernel_type` 向候选中加入一个 kernel type，不构造任何 entry；`register_kernel_builder` 登记的 builder 直接构造 target 的 kernel，不经过 kernel type 的选择，见[第 4 节](#target)。

子类通过 `self.kernel_for(interface, call)` 取得 kernel：

- 参数的写法见[添加一个新 op § kernel_for](../../new-op.md#kernel-selection)；
- kernel 接口、实现、call spec 与 entry 的定义见 [op 如何选择 kernel § 术语](../dispatch/index.md#terms)。

`kernel_for` 依次完成：

1. **确认接口。**
    - `interface` 不在 `interfaces` 中时，抛出 `OpNotAvailableError`。
    - call spec 没有给出设备且 CUDA 可用时，基类填入当前 CUDA 设备，使缓存键带有设备。
1. **按调用查找。**
    - `_entries_by_call` 以 `(interface, call)` 为键。
    - 相等的 call spec 表示同一种调用，命中只需一次查表，不读取设备属性。
1. **检查 call spec。** 未命中时，以下任一不满足就抛出 `TypeError`：
    - call spec 是接口 `request` 声明的类型；
    - 每个字段都能作为缓存键；
    - 不自行给出设备事实。
1. **按构造查找。**
    - 派发机制选出一个实现，见 [op 如何选择 kernel § 一次调用如何找到 kernel](../dispatch/index.md#call-path)；`key_for(interface, call)` 返回这一步选中的 key。
    - 实现的 `entry_for(call)` 给出 identity 与构造函数。
    - `_built_entries` 按接口分开，在每个接口下以 `(实现类, identity)` 为键：identity 相同的调用共用一个 entry，不同则构造新的 entry 并保存。
    - call spec 的设备是 CUDA 设备时，构造 entry 与处理调优请求都在 `torch.cuda.device(call.device)` 下进行。

补充说明：

- **dtype 与 entry：** 不同 dtype 的 call spec 分别派发；是否复用已构造的 entry，由选中实现的 `entry_for` 返回的 identity 决定。
- **identity 只有实现知道：** 子类不定义 identity，也不维护自己的缓存。
- **entry 只由基类持有：** 子类在本体中用局部变量接收 `kernel_for` 返回的 entry，不保存到实例上；第二次调用再经 `kernel_for` 取得，命中缓存只需一次查表。

## 6. 失败与撤销 {#failure}

调用中抛出 `Exception` 时：

- 调用栈弹出；
- 这次调用不成为 `last_call`，上一次的记录保持不变。

如果 target 是在这次调用的第 4 步选定的：

- `_reset_binding` 把绑定组重置为 `Op.__init__` 建立时的初值；
- 这次调用构造的 kernel 一并丢弃；
- 对持有的子 op 递归执行同样的处理。

处理之后，绑定组回到尚未选定 target 的状态；已完成调用的记录和持有的子 op 保留。下一次调用重新选择 target。

**表 6** 失败后的实例状态

| No. | 情况 | target | 已构造的 kernel | `last_call` |
| --- | --- | --- | --- | --- |
| 1 | target 在这次调用中选定 | 撤销 | 丢弃 | 不变 |
| 2 | target 在之前的调用中选定 | 保留 | 保留 | 不变 |

另外：

- 已经写入张量的内容不会撤销；
- `KeyboardInterrupt` 等不继承 `Exception` 的异常不经过上述处理。
