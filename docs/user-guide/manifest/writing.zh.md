# 写一个 spec

大多数 spec 只用到以下字段，本页说明它们的写法：

1. `forall`；
1. 张量的 `shape` 与 `dtype`；
1. 构造参数；
1. 简单的 `shape_rules`；
1. workload 行；
1. 内联的 roofline 公式。

可选输入、随参数变化的形状、副作用、metadata 张量等情况的写法见[扩展写法](extensions.md)。

## 1. spec 的结构 {#anatomy}

SiluAndMul 的 spec 只用到了上述字段：

```yaml
SiluAndMulFwdOp:
  family: elementwise
  status: implemented
  signature:
    forall: {M: Dim, N: Dim, T: "DType[float16 | bfloat16 | float32]"}
    inputs:
      x: {dtype: T, shape: "[M, 2 * N]"}
    outputs:
      output: {dtype: T, shape: "[M, N]"}
  workloads:
  - {M: 2048, N: 14336, dtype_cases: [{T: float16}, {T: bfloat16}], label: llama-8b-ffn-prefill}
  - {M: 1, N: 14336, dtype_cases: [{T: bfloat16}], label: llama-8b-ffn-decode}
  roofline:
    flops: "6 * M * N"
```

下面是 spec 可以包含的全部顶层字段，注释中标出了说明该字段的小节：

```yaml
<Op>:
  family: <模块名>                   # 2
  status: implemented | spec-only    # 2
  ref_api: <完整限定名>              # 2，可选
  signature:
    forall: {<index>: <kind>}        # 3
    params: {<p>: {type, default, kw_only}}                     # 4
    inputs: {<t>: {dtype, shape, optional, mutated, values, requires, ...}}  # 5
    outputs: {<t>: {dtype, shape, nullable, buffer, alias, ...}}   # 5
    types: {<Family>: {params, match, cases}}                   # 扩展写法 2
    let: {<name>: "<表达式>"}                                    # 扩展写法 4
    shape_rules: ["<refinement>"]                               # 6
    dtype_combos: [{<DType index>: <dtype>}]                    # 扩展写法 8
  workloads: [{<参数与 index>, some, dtype_cases, label}]       # 8
  roofline: {flops, bytes} | {func}                             # 9
  composition: {kind: composite, stages}                        # 扩展写法 10
```

## 2. 顶层字段 {#top}

`family`、`status`、`signature`、`workloads` 与 `roofline` 是必填字段，`workloads` 至少包含一条行。

- `family` 是 op 所在的公开模块名。op 可以通过 `tileops.<family>` 按类名导入，该模块的 `__all__` 与 manifest 保持一致。spec 所在的 YAML 文件名也由 `family` 决定，见[读写 manifest 6](index.md#layout)。
- `status` 的取值有两种：
  - `implemented` 表示已有符合 spec 的实现；
  - `spec-only` 表示尚无符合 spec 的实现，代码可能不存在，也可能只完成了一部分。依赖代码的检查只对 `spec-only` 的 op 跳过。
- `status` 只决定哪些依赖代码的检查会运行，不影响由签名生成的方法：只要一个类有对应的 spec，它就会得到全部生成的方法，包括 `eval_roofline()`。
- `ref_api` 为可选字段，记录 op 在语义上所参照的 API 的完整限定名，例如 `torch.matmul`。validator 检查它的格式；在对应模块可以导入时，还会检查该名字确实存在。

## 3. forall 与 kind {#forall}

`forall` 声明签名中所有自由的 type index 及其 kind，例如 `forall: {M: Dim, N: Dim, K: Dim, T: "DType[float16 | bfloat16]"}`。

**表 1** `forall` 中可用的 kind

| No. | kind | 取值 | 调用时如何确定 | workload 行中的写法 |
| --- | --- | --- | --- | --- |
| 1 | `Dim` | 非负整数，表示轴长 | 由输入的 unification 求得 | 整数 |
| 2 | `Shape` | 由 `Dim` 组成的元组 | 由输入的 unification 求得 | 整数列表 |
| 3 | `DType[a \| b]` | 所列 dtype 中的一个 | 由输入的 unification 求得 | `dtype_cases` |
| 4 | `Seq[Int]` | 整数列表，例如 `q_lens` | 只在实例化时存在，作为 generator 的实参（见[扩展写法 7](extensions.md#generators)） | 整数列表，或返回列表的值 primitive 调用，例如 `"repeat(512, 64)"` |

- 要求 `Int` 的位置可以使用 `Dim`，要求 `Seq[Int]` 的位置可以使用 `Shape`。
- 形状中的每个轴都是一个整数表达式。kind 不是 `Dim` 的轴，以及以 `*p` 展开的序列中的每个元素，都必须是非负数；生成的检查会在构造时或调用时确认这一点。
- YAML 中的值按声明的 kind 或 `type` 转换为 Python 值：`Shape` 与 `tuple[...]` 转为 tuple，`Seq[Int]` 与 `list[...]` 保持为 list，dtype 名转为 `torch.dtype`，ADT 值转为对应的 Python 对象。
- 在同一个 spec 中，index、`let` 与张量的名字互不相同。`out` 是保留名，不能用作张量、参数或 index 的名字。

## 4. params {#params}

`signature.params` 与 `__init__` 的参数表一一对应。普通的构造参数声明 `type`，以及可选的 `default` 与 `kw_only`（表示只能以关键字方式传入）；构造时传入的张量改为声明 `dtype` 与 `shape`，见[扩展写法 5](extensions.md#placement)。

- 以 spec 为准。对于 `implemented` 的 op，validator 逐项比较 `params` 与 `__init__`，要求参数集合、顺序、`default` 与 `kw_only` 都相同。
- 代码中可以额外出现的只有两类参数：所有 op 共有的执行策略参数 `target`，以 `*, target=None` 写在 manifest 参数之后；以及类属性 `injected_parameters` 列出的、由调用方注入的实现对象。
- 签名中的调用期输入与输出缓冲，按顺序构成 `forward` 参数表的开头部分。`forward` 可以在其后追加由代码定义的执行参数，这些参数不属于签名。

构造参数可以直接出现在类型中，不需要额外标注：

- `int` 参数可以写在形状中，例如 MoE 的 `num_local_experts`；
- `list[int]` 或 `tuple[int, ...]` 参数可以以 `*p` 的形式展开在形状中，例如 RMSNorm 的 `normalized_shape`；
- dtype 参数（`type` 为若干 dtype 名的并集）可以直接作为张量的 `dtype`，例如 Alibi 的 `out_dtype`；
- `int | None` 参数的写法见[扩展写法 1](extensions.md#presence)。

`type` 到 kind 的完整映射见设计文档 [Manifest 表 3](../../design/manifest.md#t-types)。`float`、取值不受限的 `str` 等类型的参数不参与类型推导，但可以出现在 `shape_rules` 与 roofline 公式中，validator 检查它们的 `type`、`default` 以及在这些表达式中的用法。

## 5. inputs 与 outputs {#tensors}

每个张量写作 `{dtype: ..., shape: "..."}`，对应类型 `Tensor[T, s]`。

- 形状写作 `"[" 轴, ... "]"`，或者写成 type family 的应用。每个轴可以是表达式、`*S` 或 `*primitive(...)`；`"[]"` 表示零维张量。
- 形状相同的张量使用同一个形状项，例如逐元素 op 的输入与输出都写作 `[*S]`。
- 每个 spec 只有一份签名，因此每次调用返回的输出名字与数量都相同。如果一个 op 的输出数量随某个开关变化，或者同一个参数既可以是标量也可以是张量，这个 op 应拆分为多个 spec。
- op 根据参数以及张量是否传入来选择实现；张量的内容只作为计算的输入，不影响实现的选择。

## 6. shape_rules {#refinement}

`shape_rules` 中的每一条都是一个 refinement，即约束 index 取值的谓词，在 unification 之后检查。

```yaml
shape_rules:
- "not is_causal or S_q <= S_kv"
- "D > 0 and D % 2 == 0"
```

- 形状、`let`、type family、refinement 与内联 roofline 公式使用同一套封闭的表达式语言，运算优先级与 Python 相同。表达式语言的组成见设计文档 [Manifest 表 10](../../design/manifest.md#t-lang)。
- refinement 只能依赖运行时可以得到的量。能在构造时求值的 refinement 在构造时检查，其余的在每次调用时检查。
- 对 metadata 张量内容的约束写在张量的 `requires` 中，而非 `shape_rules` 中，见[扩展写法 7](extensions.md#generators)。
- 条件判断会在其所选分支内收窄 kind：`present(v)` 为 true 的分支中，`Maybe[X]` 收窄为 `X`；`x == 'a'` 或 `x in ('a', 'b')` 成立的分支中，`x` 的取值收窄为这些字面量。如果一个比较所用的字面量不是对应枚举或 dtype 集合中的任何成员，validator 会拒绝这条 refinement。
- refinement 是否可以满足，由 spec 的作者负责。
- `x.shape == (...)`、`x is None`、`isinstance` 等写法不被接受，对应的改写方式见[调用与校验 7](calls.md#rejected)。

## 7. dtype {#dtype}

张量的 `dtype` 是一个 dtype 表达式，有以下四种形式：

**表 2** dtype 表达式的形式

| No. | 形式 | 含义 | 例子 |
| --- | --- | --- | --- |
| 1 | `forall` 中的 `DType` index | 在声明的集合内取值，由输入的 unification 求得 | `a: {dtype: T}` |
| 2 | dtype 参数 | 取构造参数的值 | `output: {dtype: out_dtype}` |
| 3 | 常量 | 固定的 dtype | `cu_seqlens_q: {dtype: int32}` |
| 4 | dtype primitive | 由其他 dtype 计算得到 | `promote_int_to_float(T)`、`coalesce_dtype(out_dtype, D)` |

如果 spec 没有 `dtype_combos`，各个 `DType` index 在各自的集合内独立取值。多个 `DType` index 只允许特定组合时的写法见[扩展写法 8](extensions.md#dtype-combos)。

## 8. workload 行 {#workloads}

每条 workload 行确定一次调用。benchmark、nightly 与 manifest 相关的测试都从 workload 行生成调用。

```yaml
workloads:
- {M: 2048, N: 14336, dtype_cases: [{T: float16}, {T: bfloat16}], label: llama-8b-ffn-prefill}
```

**表 3** workload 行的键

| No. | 键 | 值 |
| --- | --- | --- |
| 1 | 构造参数名 | 参数的值；没有默认值的参数必须给出 |
| 2 | `some` | 本次调用传入的可选张量，见[扩展写法 1](extensions.md#presence) |
| 3 | `forall` 中的 `Dim`、`Shape`、`Seq[Int]` index | 当前分支用到、且不由 generator 求得的 index |
| 4 | `dtype_cases` | 一个非空列表，每一项是当前分支用到的 `DType` index 的一组取值，例如 `[{T: float16}, {T: bfloat16}]`；只在有这类 index 时写；dtype 参数按构造参数给出 |
| 5 | `label` | 这条行的名字 |

- 一个 index 在某个分支上被用到，是指它出现在该分支的以下位置之一：形状、dtype、refinement、generator 实参、`requires`、内联 roofline 公式。workload 行必须给出当前分支用到的每一个 index，且只给出这些 index；`let` 与由 generator 求得的 index 不在行中给出。
  - 判断时，各处表达式先按当前分支化简。如果一条 refinement 的条件在该分支上恒为 true，其中出现的 index 不算被用到。
  - 一个 `let` 被用到时，它的表达式中出现的 index 也算被用到。
  - `func` 形式的 roofline 不会使任何 index 被用到。
  - 选择 type family 分支、决定张量是否存在或决定输出是否为 `None` 的 discriminant，总是算作被用到。
- 每条行按 `dtype_cases` 展开为若干个 case。case id 由以下三部分依次以 `-` 连接而成：

  1. `label`；
  1. `dtype_cases` 中的各 dtype 值，按 `forall` 的声明顺序；
  1. dtype 参数的值，按 `params` 的声明顺序。

  例如上例第一个 case 的 id 是 `llama-8b-ffn-prefill-float16`。
- nightly 的历史数据以 case id 为键，因此修改 `label` 会使这条行的历史记录中断。
- `label` 不能为空，长度不超过 24 个字符，只能包含 `[A-Za-z0-9._-]` 中的字符。`label` 只描述这条行所建模的场景，即模型及其用途或合成目的，再加上区分同类行所需的限定词，例如 `llama-8b-ffn-prefill`；op 名与 dtype 已经出现在 case id 中，不必重复。共用同一个 `label` 的行只能在 dtype 上不同。
- 同一个 spec 内的 case id 互不相同。
- 实例化时，workload 行确定形状、dtype、参数取值、可选张量是否传入，以及 metadata 张量的取值。其余部分按固定规则生成：设备的确定方式见[调用与校验 4](calls.md#device)，stride 连续，张量之间没有别名，普通数据随机生成。
- validator 会根据实例化后的输入重新推断这次调用，并要求结果与 workload 行一致。
- workload 行不承担单元测试的覆盖职责，覆盖 kernel 各分支所需的形状由各 op 的测试另行选取。

## 9. roofline {#roofline}

`roofline` 给出一次调用的 FLOPs 与字节数，有内联公式与 `func` 两种写法：

```yaml
roofline:
  flops: "2 * M * N * K"        # 内联公式；省略 bytes 时由签名推导
# 或
roofline:
  func: "tileops.perf.formulas.gqa_fwd_roofline"
```

- 内联公式使用表达式语言书写，可以引用签名中的 index、构造参数与 `let`，以及 `present(t)`、`bytes(t)`（张量 `t` 的字节数）和内建 primitive。如果代价取决于某个张量是否传入，公式可以使用 `present(...)` 区分。
- 省略 `bytes` 时，字节数按每个张量完整读写一次推导：
  - 未被写入的输入计一次读，输出计一次写，`mutated` 的输入读写各计一次，`write_only` 的输入只计一次写；
  - 签名中的每个张量名按一块独立的存储计算，即使调用方把同一个张量传给两个参数；只有签名中声明的别名（`buffer`、`alias`）视为同一块存储；
  - 每个张量的字节数为 `prod(shape) * bits(dtype) / 8`，打包 dtype 按载体计算。
- 字节数表示算法的最少访存量：中间结果不计入，算法只读取一部分的输入按实际读取的不同元素计数；两者在每条 workload 行上相差不到 1% 时，可以按整个张量计。
- 如果推导出的字节数与上述最少访存量不符，spec 需要显式写出 `bytes`，并附带相应的测试。
- `flops` 是算法完成本次调用所需的最少算术量，不是硬件实际执行的指令数：已经算出的值按复用计；linear attention、状态空间扫描等循环 op，签名中带 chunk size 时按该 chunk size 的分块算法计，否则按逐 token 递推计；逐元素运算、attention 与 MoE 按设计文档 [Roofline 1.3](../../design/roofline.md#13-convention) 的约定计数；在本次调用的 dtype 上等同于恒等映射的路径计 0。
- 公式需要 Python 逻辑时使用 `func`。它指向 `tileops.perf.formulas` 中的一个模块级函数 `f(call) -> tuple[int, int]`。函数只读取参数 `call`，不读取 op 实例。`call` 是一次经过检查的调用，提供以下接口：

  **表 4** roofline `func` 的参数 `call`

  | No. | 接口 | 内容 |
  | --- | --- | --- |
  | 1 | `call.indices` | 参数、本次调用求得的 index 与 dtype index、用到的 `let`，即内联公式可以引用的名字 |
  | 2 | `call.present(t)` | 张量 `t` 是否传入、持有或返回；`call.present("out")` 表示调用方是否传入了 `out` |
  | 3 | `call.tensors[t]` | 张量 `t` 的 `(形状, dtype 名)` |
  | 4 | `call.bytes(t)` | 张量 `t` 的字节数 |
  | 5 | `call.metadata_values(t)` | metadata 张量 `t` 的内容；在 meta 张量上调用时报错，因为 meta 张量没有取值 |
  | 6 | `call.stages` | 复合 op 的各子 op 在本次调用中完成的调用，按 stage 名索引 |

- 每个 spec 都会生成一个 `eval_roofline()` 方法，它基于 op 最近一次完成的调用计算 FLOPs 与字节数。benchmark 通过这个方法取得数值并写入结果，roofline 工具读取 benchmark 的结果，不直接调用 op。

roofline 的完整规则见设计文档 [Roofline](../../design/roofline.md)。
