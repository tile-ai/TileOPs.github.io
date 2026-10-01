# 概念

spec 的核心是签名。签名在形式上是一个 polymorphic function type。本页以 GEMM 为例，依次介绍描述签名所用的概念；各字段的具体写法见[写一个 spec](writing.md)与[扩展写法](extensions.md)。

## 1. polymorphic function type {#pft}

GEMM 的签名写成数学形式如下：

```
Mat[t: Bool, R: Dim, C: Dim] = match t { false → [R, C]; true → [C, R] }

gemm(trans_a, trans_b : Bool) : ∀ (M N K : Dim) (T : DType[float16 | bfloat16]).
       Tensor[T, Mat[trans_a, M, K]] → Tensor[T, Mat[trans_b, K, N]] → Tensor[T, [M, N]]
```

对应的 YAML 见[示例 1](examples.md#gemm)。

函数名之后的 `trans_a`、`trans_b` 是构造参数，`∀` 之后的 `M`、`N`、`K`、`T` 是被量化的名字。这个签名由三层组成：

**表 1** 签名的三层结构

| No. | 层次 | 数学形式 | manifest 中的写法 |
| --- | --- | --- | --- |
| 1 | 张量类型 | `Tensor[T, s]`，其中 `T` 是 dtype，`s` 是形状 | `{dtype: T, shape: "[M, N]"}` |
| 2 | 函数类型 | 从输入张量的类型到输出张量的类型 | `inputs`、`outputs` |
| 3 | quantification | `∀` 列出所有 type index 及其 kind | `forall` |

对 type index 做了 quantification 的函数类型称为 polymorphic function type。运行时检查与 `eval_roofline()` 都从签名生成；声明了 compile boundary 的 op，其 `torch.library` operator 与 fake/meta 函数也从签名生成。validator 以签名为对象做静态检查。

## 2. type index 与 kind {#index}

type index 是类型中的参数名，例如 GEMM 的 `T`、`M`、`N`、`K`。每个 type index 都有一个 kind，kind 规定了它的取值范围：

**表 2** 常用的 kind

| No. | kind | 取值 | 例子 |
| --- | --- | --- | --- |
| 1 | `Dim` | 非负整数，表示轴长 | GEMM 的 `M`、`N`、`K` |
| 2 | `Shape` | 由 `Dim` 组成的元组，用于秩不固定的张量 | 逐元素 op 的形状 `[*S]` |
| 3 | `DType[...]` | 所列 dtype 中的一个 | GEMM 的 `T` |
| 4 | `Bool`、枚举、ADT | 有限个取值 | GEMM 的 `trans_a`、`trans_b` |

签名中每个 type index 的来源是唯一的，来源有三种：

- 在 `forall` 中声明，例如 `M`、`T`；
- 来自构造参数，其 kind 由参数的 `type` 决定，例如 `trans_a` 的 `type` 是 `bool`，对应的 kind 是 `Bool`；
- 由 `let` 定义，见[第 5 节](#let)。

## 3. type family {#family}

type family 是一个根据 discriminant 的取值给出形状的函数。discriminant 指取值有限的量，例如 `Bool` 参数、枚举参数、ADT 参数，以及某个张量是否传入。

GEMM 中的 `Mat` 就是一个 type family：`t` 为 false 时结果是 `[R, C]`，为 true 时结果是 `[C, R]`。张量 `a` 的形状写作 `Mat[trans_a, M, K]`，于是：

- 构造时 `trans_a` 的值已经确定，`Mat` 选定分支，`a` 的形状随之确定为 `[M, K]` 或 `[K, M]`；
- 调用时，`M` 与 `K` 的值通过 unification 从 `a` 的实际形状中求得。

type family 定义在使用它的 spec 的 `signature.types` 中。

## 4. ADT {#adt}

ADT（algebraic data type）是带 tag 的 sum type：一个 ADT 值属于若干 constructor 中的一个，每个 constructor 有各自的字段。

MoE staged 系列 op 的 layout 参数是 ADT `MGroupedLayout`，它有两个 constructor：

**表 3** `MGroupedLayout` 的 constructor

| No. | constructor | 字段 | 含义 |
| --- | --- | --- | --- |
| 1 | `contiguous` | `packing`、`metadata_kind`、`alignment` | 各专家的行连续排放，`packing` 决定是否按 `alignment` 对齐 |
| 2 | `masked` | `max_m` | 每个专家固定占 `max_m` 行，多出的行由 mask 标记 |

- `masked` 值只有 `max_m` 一个字段，`contiguous` 值只有另外三个字段。
- type family 可以按 constructor 做 pattern matching，并在 `masked` 分支中读取 `layout.max_m`。
- 每个 constructor 对应一个 Python 类，例如 `contiguous` 对应 `ContiguousLayoutSpec`。
- 多个 spec 共用的 ADT 定义在 `spec/types.yaml` 中。

`MGroupedLayout` 的定义以及使用它的 spec 见[示例 2](examples.md#moe)。

## 5. `let` {#let}

`let` 为由 index 计算得到的量命名。以 MaxPool2d 的输出高度为例：

```yaml
let:
  kH: "per_axis(kernel_size, 0, 2)"
  H_out: "pool.out(H_in, kH, sH, pH, dH, ceil_mode)"
```

`H_out` 由输入高度 `H_in` 与构造参数计算得到，并用于输出形状。`let` 的值由签名计算，workload 行中不需要给出。

## 6. refinement {#refinement}

refinement 是约束 index 取值的谓词，写在 `shape_rules` 中。以 attention 为例：

- 类型 `q: [B, S_q, H, D]` 与 `k: [B, S_kv, H, D]` 本身不限制 `S_q` 与 `S_kv` 的大小关系；
- refinement `not is_causal or S_q <= S_kv` 表示在 causal 情形下要求 `S_q <= S_kv`。

## 7. unification 与 type inference {#unification}

unification 将声明的形状与实际张量的形状逐轴对应，求出未知的 index，并检查同一个 index 在各处的取值是否一致。以 `trans_a = trans_b = false` 时的一次 GEMM 调用为例：

```
声明：  a: Tensor[T, [M, K]]      b: Tensor[T, [K, N]]
实际：  a: bfloat16, (128, 4096)   b: bfloat16, (4096, 512)
```

**表 4** 这次 GEMM 调用的 unification 过程

| No. | 对应关系 | 结果 |
| --- | --- | --- |
| 1 | `a` 的 dtype 对应 `T` | `T := bfloat16` |
| 2 | `a` 的第 1 轴对应 `M` | `M := 128` |
| 3 | `a` 的第 2 轴对应 `K` | `K := 4096` |
| 4 | `b` 的 dtype 对应 `T` | 检查其与 `T` 相等 |
| 5 | `b` 的第 1 轴对应 `K` | 检查其与 `K` 相等 |
| 6 | `b` 的第 2 轴对应 `N` | `N := 512` |

所有 index 求出之后，输出类型 `Tensor[T, [M, N]]` 即确定为 `bfloat16, (128, 512)`。

type inference 指调用时求出全部 index 的过程，unification 是其中处理等式的部分；index 之间的其他关系由 refinement 检查。哪些轴的写法可以通过 unification 求解，见[调用与校验 3](calls.md#inference)。

## 8. effect {#effect}

effect 描述一次调用对参数的写入以及输出与输入之间的别名关系。以激活函数的 `inplace` 参数为例，当它为 true 时，op 写入输入张量，并将这个输入对象作为输出返回：

```yaml
inputs:  {input: {dtype: T, shape: "[*S]", mutated: inplace}}
outputs: {output: {dtype: T, shape: "[*S]", alias: input}}
```

effect 决定了 roofline 的读写计数，以及声明了 compile boundary 的 op 所生成的 operator schema（其中的 `mutates_args`）。所有 effect 声明见[扩展写法 6](extensions.md#effects)。

## 9. 术语表 {#glossary}

**表 5** 术语的含义与 manifest 中的写法

| No. | 术语 | 含义 | manifest 中的写法 | 所在小节 |
| --- | --- | --- | --- | --- |
| 1 | `Tensor[T, s]` | 以 dtype `T` 与形状 `s` 为参数的张量类型 | 张量的 `dtype`、`shape` | [1](#pft) |
| 2 | polymorphic function type | 对 type index 做了 quantification 的函数类型 | `forall` | [1](#pft) |
| 3 | type index | 类型中的参数名 | `forall`、构造参数、`let` | [2](#index) |
| 4 | kind | type index 的种类，决定其取值范围与可用的运算 | `forall: {M: Dim}` | [2](#index) |
| 5 | discriminant | 取值有限的量，用于选择 type family 的分支，或决定张量是否存在 | `match`、`optional`、`nullable`、`mutated` 中的表达式 | [3](#family) |
| 6 | type family | 根据 discriminant 的取值给出形状的函数 | `signature.types` | [3](#family) |
| 7 | ADT、constructor | 带 tag 的 sum type，每个 constructor 有各自的字段 | `spec/types.yaml` 中的 `adts` | [4](#adt) |
| 8 | `let` | 由 index 计算得到的具名量 | `let: {H_out: "..."}` | [5](#let) |
| 9 | refinement | 约束 index 取值的谓词 | `shape_rules` 中的每一条 | [6](#refinement) |
| 10 | unification | 将声明的形状与实际形状逐轴对应，求出未知的 index | 由签名生成 | [7](#unification) |
| 11 | type inference | 调用时求出全部 index 的过程 | 由签名生成 | [7](#unification) |
| 12 | effect | 调用对参数的写入与别名关系 | `mutated`、`write_only`、`buffer`、`alias` | [8](#effect) |
| 13 | `Maybe[X]`、`present` | 可能缺省的 `X`；`present(x)` 表示它是否给出 | 可选输入、可空输出、`int \| None` 参数 | [扩展写法 1](extensions.md#presence) |
| 14 | primitive | 表达式中可以调用的内建函数，例如 `broadcast`、`pool.out` | 表达式中的函数调用 | [扩展写法 4](extensions.md#let) |
| 15 | generator | 实例化时为 metadata 张量生成取值的函数，例如 `prefix_sum(q_lens)` | 张量的 `values` | [扩展写法 7](extensions.md#generators) |
