# 扩展写法

本页介绍只有部分 spec 才会用到的写法。每一节对应一种情况，各节可以独立阅读。大多数 spec 所需的字段见[写一个 spec](writing.md)。

## 1. 实参类型：可选输入、可空输出与可缺省参数 {#presence}

调用期输入像构造参数一样声明实参的 `type` 与 `default`。`type` 列出这个实参可以持有的成员。以 FusedMoESharedExpert 为例：

```yaml
inputs:
  correction_bias:  {type: "Tensor | None", default: null, dtype: float32, shape: "[E]"}
  shared_w_gate_up: {type: "Tensor | None", default: null, dtype: D, shape: "[2 * S, H]"}
  shared_w_down:    {type: "Tensor | None", default: null, dtype: D, shape: "[H, S]"}
outputs:
  shared_output: {dtype: D, shape: "[T, H]", nullable: "present(shared_w_gate_up)"}
  routed_output: {dtype: D, shape: "[T, H]"}
shape_rules:
- "present(shared_w_down) == present(shared_w_gate_up)"
```

**表 1** 实参类型的成员

| No. | 成员 | 接受的值 | 收窄为该成员的分支上读作 |
| --- | --- | --- | --- |
| 1 | `Tensor` | 张量 | 该声明的 `Tensor[T, s]` |
| 2 | `list[Tensor]` | 由张量组成的 list 或 tuple | 张量序列，见[第 11 节](#sequences) |
| 3 | `int` | 不是 `bool` 的 `int` | `Int`，调用期标量 |
| 4 | `float` | `float`；类型中没有 `int` 和 `Number` 成员时，也接受 `int`（`bool` 除外） | 一个值，调用期标量 |
| 5 | `Number` | `int` 或 `float` | 一个值，调用期标量 |
| 6 | `bool` | `bool` | `Bool`，调用期标量 |
| 7 | `None` | `None` | `None` |

实参类型的规则：

- `type` 默认为 `Tensor`，表示该声明的 `Tensor[dtype, shape]`。`Number` 已经包含 `int` 与 `float`，因此不与它们同时列出。
- 有 `Tensor` 或 `list[Tensor]` 成员的实参是调用期输入，声明在 `inputs` 中；其他实参都是构造参数，声明在 `params` 中。张量不作为构造参数。
- 值按它的 Python 类型对应成员：`None` 对应 `None`；list 或 tuple 对应 `list[Tensor]`；张量对应 `Tensor`；`bool` 只对应 `bool`；`int` 对应 `int`，没有 `int` 成员时对应 `Number`，两者都没有时对应 `float`；`float` 对应 `float`，没有 `float` 成员时对应 `Number`。没有成员可以对应的值被拒绝。
- 没有 `default` 的输入每次调用都要传入。有 `default` 的输入可以省略：`forward` 以 `default` 作为它的默认值，省略这个输入与传入该值效果相同。
- 在按位置传入的输入中，没有 `default` 的输入不排在有 `default` 的输入之后。`kw_only` 的输入排在所有按位置传入的输入之后，也可以是必须传入的。
- 必须传入、但可以为 `None` 的实参声明 `None` 成员，不声明 `default`，例如 `bias: {type: "Tensor | None", dtype: Wb, shape: "[N]"}`。
- `member(x)` 是 `x` 所持有成员的名字，例如 `'Tensor'` 或 `'Number'`；每个有多个成员的实参都是一个取值有限的 discriminant。`present(x)` 表示 `member(x) != 'None'`，无论 `x` 是调用方传入的还是取了 `default`。
- `member(x) == '<m>'` 把 `x` 收窄为该成员，`member(x) != '<m>'` 排除该成员，`present(x)` 排除 `None`。收窄可以经由 `not`、`and`、`or` 与条件表达式组合。
- 张量声明 `shape`、`dtype`、`contiguous`、`device`、`values`、`requires` 与 effect 只在实参持有张量成员的分支上生效。只被它们用到的 index 在其他分支上不算被用到：在上例中，不传 `shared_w_gate_up` 时 `S` 没有被用到，对应的 workload 行不需要给出 `S`。

必须同时传入或同时不传的几个输入，各自声明 `type: "Tensor | None", default: null`，再由一条只读取 discriminant 的 refinement 把它们关联起来，例如上例中的 `present(shared_w_down) == present(shared_w_gate_up)`。这样的 refinement 是定义域限制，见[第 2 节](#type-family)。

可能返回 `None` 的输出使用 `nullable: "<表达式>"` 声明。在上例中，传入 `shared_w_gate_up` 时 op 返回 `shared_output`，否则该输出为 `None`。表达式只能由取值有限的 discriminant 组成，包括 `Bool` 与枚举参数、ADT 的 tag 与取值有限的字段、`present(...)` 与 `member(...)`。

`int | None` 参数的 kind 是 `Maybe[Int]`。`present(v)` 表示它是否给出；在 `present(v)` 成立的分支上，`v` 本身就是它的值，例如 `let` 中的 `R: "rotary_dim if present(rotary_dim) else D"`。

**调用期标量。** 收窄为标量成员的实参是调用期标量。RoPECosSin 的 `seqlen_offsets` 可以是所有序列共用的一个偏移，也可以是每个序列一个偏移的张量：

```yaml
inputs:
  seqlen_offsets: {type: "int | Tensor", default: 0, dtype: O, shape: "[B]"}
shape_rules:
- "member(seqlen_offsets) != 'int' or 0 <= seqlen_offsets <= P - L"
```

- 调用期标量的取值只在 refinement 与 `let` 中读取；直接或经由其他 `let` 读取它的 `let`，同样只在 refinement 与这类 `let` 中读取。
- 形状、dtype、type family、`nullable` 与 `mutated` 的条件、`values`、`requires` 与 roofline 公式都不读取调用期标量的取值，validator 拒绝这样读取的 spec。因此取值不决定任何输出形状，也不决定任何分支。构造参数不受这条限制。
- 读取调用期标量的 refinement 与 `let` 每次调用都执行。Op plan 以每个实参持有的成员与标量的类型为键，不以标量的取值为键。

**workload 行。**

- `some` 列出在这条行中传入张量成员的、有多个成员的实参。每个传入的输入都要列出，同时传入的几个输入也逐一列出：传入共享专家写作 `some: [shared_w_gate_up, shared_w_down]`。
- 其他有多个成员的实参取这条行给出的标量值或 `null`，没有给出时取 `default`。例如 Clamp 的行 `{A: [4096, 4096], min: -0.5, max: 0.5, ...}` 以数值传入两个界。必须传入、但可以为 `None` 的实参写 `null` 表示传入 `None`。
- 对于 `implemented` 的 op，每个有多个成员的实参中被定义域限制接受的每个成员，都要至少由一条行取到。

**operator schema。** 对于有编译边界的 op，每个输入成为生成的 `torch.library` operator 的一个或多个 slot：

**表 2** 输入对应的 operator slot

| No. | 成员 | slot 类型 | slot 名 |
| --- | --- | --- | --- |
| 1 | `Tensor`，唯一的成员 | `Tensor` | `<name>` |
| 2 | `list[Tensor]` | `Tensor[]` | `<name>` |
| 3 | `Tensor`，多个成员之一 | `Tensor?` | `<name>` |
| 4 | `int` | `SymInt?` | `<name>__int` |
| 5 | `float` | `float?` | `<name>__float` |
| 6 | `Number` | `Scalar?` | `<name>__number` |
| 7 | `bool` | `bool?` | `<name>__bool` |
| 8 | `None` | 没有 slot | — |

- 只有一个成员的实参占一个 slot，名字就是实参名。有多个成员的实参，每个 `None` 以外的成员占一个可为空的 slot。
- 一次调用至多传入一个实参的一个 slot；只有 `None` 是成员时才可以一个 slot 都不传。operator 与它的 fake 函数都检查这一点。
- slot 名与其他实参名或执行参数名相同时被拒绝。

## 2. 随参数变化的形状：type family {#type-family}

当参数决定张量的秩或轴的顺序时，形状使用 type family 描述。type family 定义在 spec 的 `signature.types` 中：

```yaml
signature:
  types:
    Mat:
      params: {t: Bool, R: Dim, C: Dim}
      match: t
      cases:
      - {when: false, is: "[R, C]"}
      - {when: true, is: "[C, R]"}
  inputs:
    a: {dtype: T, shape: "Mat[trans_a, M, K]"}   # 实参依次对应 Mat 的 t、R、C
```

- 张量的 `shape` 写作 `<Family>[实参, ...]`，实参按 `params` 的声明顺序对应。
- `match` 的对象必须是取值有限的 discriminant，即 `Bool`、枚举、ADT、`present(...)`、`member(...)`、由它们构成的布尔表达式（例如 `member(min) == 'Tensor'`），或由它们组成的元组。对元组做 match 时，`when` 写成列表，例如 `when: [true, false]`。
- `Maybe[ADT]` 类型的 discriminant 按它的值匹配；只有 `_` 能匹配它缺省的情况。
- `cases` 在 spec 接受的所有取值上必须既无遗漏也无重叠。
- 多个张量应用同一个 type family 时，它们总是取同一个分支。
- type family 之间的引用不能成环；如果某个 type family 没有被任何形状使用，validator 会报错。

Clamp 是按实参成员做 match 的例子。`min` 与 `max` 各自可以持有数值、张量或 `None`，只有张量形式的界参与输出形状的广播：

```yaml
types:
  ClampOut:
    params: {A: Shape, L: Shape, U: Shape, tl: Bool, tu: Bool}
    match: [tl, tu]
    cases:
    - {when: [true, true],   is: "[*broadcast(A, L, U)]"}
    - {when: [true, false],  is: "[*broadcast(A, L)]"}
    - {when: [false, true],  is: "[*broadcast(A, U)]"}
    - {when: [false, false], is: "[*A]"}
inputs:
  input: {dtype: T, shape: "[*A]"}
  min: {type: "Number | Tensor | None", default: null, dtype: T, shape: "[*L]"}
  max: {type: "Number | Tensor | None", default: null, dtype: T, shape: "[*U]"}
outputs:
  output: {dtype: T, shape: "ClampOut[A, L, U, member(min) == 'Tensor', member(max) == 'Tensor']"}
```

只读取 discriminant 的 refinement 称为**定义域限制**。定义域限制在选择 type family 分支之前检查，被它排除的取值不需要对应的 case。以 MoEPrePermute 为例：

```yaml
types:
  LayoutMetadata:
    params: {layout: MGroupedLayout, E: Dim, P: Dim}
    match: layout
    cases:
    - {when: {contiguous: {metadata_kind: physical_psum}}, is: "[E]"}
    - {when: {contiguous: {metadata_kind: per_row}}, is: "[P]"}
shape_rules:
- "layout.kind == 'contiguous'"
```

`layout.kind == 'contiguous'` 只读取 discriminant `layout`，它排除了 masked layout，因此两个 case 已经覆盖了 MoEPrePermute 接受的所有取值。判断一条 refinement 是否只读取 discriminant 时，考察的是它出现的所有名字，与操作数的书写顺序无关。Clamp 要求至少传入一个界的 `present(min) or present(max)` 也是定义域限制。

## 3. 带字段的参数：ADT {#adt}

当一个参数有有限的几种形态，且每种形态带有各自的字段时，参数的类型使用 ADT 描述。ADT 定义在 `spec/types.yaml` 中，供多个 spec 共用：

```yaml
adts:
  MGroupedLayout:
    sum:
      contiguous:
        python: tileops._ops.moe.contracts.ContiguousLayoutSpec
        fields:
          packing: {type: "'tight' | 'aligned'", python: tileops._ops.moe.contracts.ContiguousPacking}
          metadata_kind: {type: "'physical_psum' | 'per_row'", python: tileops._ops.moe.contracts.ContiguousMetadata}
          alignment: Dim
        invariant: "(packing == 'tight') == (alignment == 1) and alignment >= 1"
      masked:
        python: tileops._ops.moe.contracts.MaskedLayoutSpec
        fields: {max_m: Dim}
```

- 每个 constructor 对应一个 Python 类，由 `python` 指定。对象的 `kind` 属性是 constructor 名，各字段是同名属性，枚举字段的取值是属性的 `.value`。
- ADT 值写作 `{constructor: {字段: 值}}`，workload 行中也使用这种写法。
- `invariant` 是 constructor 上可选的 refinement，在实例化与构造时都会检查。
- type family 可以按 constructor 做 pattern matching：`{masked: _}` 匹配任意 masked 值，`{contiguous: {metadata_kind: per_row}}` 同时约束了一个取值有限的字段。constructor 特有的字段只能在匹配了该 constructor 的分支中读取，例如 masked 分支中的 `layout.max_m`。
- ADT 是 sealed 的，constructor 与字段在定义处确定。为共用的 ADT 增加 constructor 时直接修改其定义；不接受新 constructor 的 spec 可以用一条定义域限制排除它，无需修改 type family。

## 4. 计算得到的量：let 与 primitive {#let}

形状或公式中需要用到由 index 计算得到的量时，这个量定义为 `let`。以 MaxPool2d 为例：

```yaml
let:
  kH: "per_axis(kernel_size, 0, 2)"
  sH: "per_axis(stride, 0, 2, fallback=kH)"
  H_out: "pool.out(H_in, kH, sH, pH, dH, ceil_mode)"
outputs:
  output: {dtype: T, shape: "[N, C, H_out, W_out]"}
```

- `let` 的值由签名计算：能在构造时求值的在构造时计算，其余的在调用时计算。workload 行中不需要给出 `let`。
- 一个 `let` 可以引用其他 `let`，例如 `sH` 引用了 `kH`，但 `let` 之间的依赖不能成环。

primitive 是表达式中可以调用的内建函数，例如 `broadcast`、`reduced`、`per_axis`、`ceil_div`。

- primitive、generator 与 `requires` 中的谓词都是固定的集合，完整列表在 `tileops.manifest.primitives` 中，每个成员的 docstring 说明了它的计算内容。
- 只供某一个 family 使用的成员带有 family 前缀，例如 `pool.out`、`moe.capacity`。
- 参数超出定义域时，primitive 会报错，错误信息指向调用它的那条声明。
- 新增成员需要修改 `tileops.manifest.primitives`，并附带测试。
- 所有接受轴参数的 primitive 都按同一规则处理轴：对零维张量，`0` 与 `-1` 都表示唯一的标量轴；其他情况下，轴的取值范围是 `[-rank, rank)`。

## 5. 内存布局与设备 {#placement}

- 每个张量都是调用期输入。由类型系统之外的值组成的列表是构造参数，只通过它的长度读取；例如 LongRoPE 的 `rescale_factors` 是 `list[float] | None` 参数，由 `not present(rescale_factors) or len(rescale_factors) == D // 2` 检查。
- 要求内存连续的张量声明 `contiguous: true`，例如 MoE staged 系列 op 的输入与输出。受支持的调用使用连续的输入（[API 稳定性](../../stability.md)）；没有这项声明的张量不连续时，不保证可用。
- 必须位于 CPU 上的张量声明 `device: cpu`，例如 `GDNFwdOp` 的 `cu_seqlens_cpu`。
- 没有调用期张量输入的 op 声明 `device` 参数，例如 Alibi。调用设备的确定方式见[调用与校验 4](calls.md#device)。

## 6. 对参数的写入：effect {#effects}

没有 effect 声明的 op 只读取输入，并为输出分配新的张量。如果 op 会写入参数，需要在签名的相应张量上声明 effect。effect 决定了 roofline 的读写计数，以及有编译边界的 op 所生成的 operator schema。

**表 3** effect 声明

| No. | 声明 | 含义 | 例子 |
| --- | --- | --- | --- |
| 1 | 输出上的 `buffer: out` | `forward` 在所有未声明 `kw_only` 的输入之后增加只能以关键字传入的参数 `out`，默认为 `None`。调用方传入 `out` 时，op 将结果写入并返回它；未传入时，op 分配新张量。`out` 与该输出的形状和 dtype 相同 | MoEGroupedGEMM 的 `output` |
| 2 | 输入上的 `mutated: true` | op 可能写入这个输入，它在调用前的内容参与计算 | |
| 3 | 输入上的 `mutated: true` 与 `write_only: true` | 必须传入的结果缓冲：op 覆盖写入，结果只取决于其他输入；如果 op 返回 `None`，`outputs` 为空 | 目前没有 op 使用 |
| 4 | 输入上的 `mutated: "<discriminant 表达式>"` | 仅当表达式为 true 时，op 才写入这个输入 | 激活函数的 `mutated: inplace` |
| 5 | 输出上的 `alias: <输入名>` | 该输入被写入时，这个输出就是该输入对象本身 | 激活函数的 `alias: input` |
| 6 | 输入 `out` 上的 `mutated: true`、`write_only: true` 与 `kw_only: true` | 无法自行分配输出的 op 所必须传入的输出缓冲：只能以关键字传入，没有默认值，位于所有输入的最后；一个输出声明 `alias: out` | PagedKVCacheGather 的 `out` |
| 7 | 被写入的输入或 `buffer: out` 输出上的 `overlaps: [<输入名>, ...]` | 被写入的张量可以与列出的输入共享字节 | MoEExpertMLPFwdOp 的 `output` |

调用的返回值由这些声明决定：

- 传入了 `out` 的 op 写入 `out`，并返回这个对象；
- 有多个输出的 op 返回一个 tuple，每个声明的输出按声明顺序占一个位置，缺省的 `nullable` 输出在其位置上为 `None`；
- 唯一的 effect 是写入输入的 op 不声明输出，返回 `None`。

调用写入的张量（`out` 或被写入的输入）不与其他调用张量共享任何字节，除非它在 `overlaps` 中列出了那个输入；违反这一条的调用在任何实现运行前抛出 `CheckError`。检查比较每个张量从首个元素到末个元素所占的字节区间，因此同一块内存上交错的两个视图也算重叠。只有每个实现都在写入该张量之前读完某个输入时，条目才列出这个输入，例如只由组合 op 的前一个 stage 读取的输入。

以激活函数的 `inplace` 为例：

```yaml
params:
  inplace: {type: bool, default: false, kw_only: true}
inputs:
  input: {dtype: T, shape: "[*S]", mutated: inplace}
outputs:
  output: {dtype: T, shape: "[*S]", alias: input}
```

effect 声明还需满足以下规则，validator 会拒绝违反它们的 spec：

- effect 只能声明在没有标量成员的输入上，不能声明在张量序列上；
- `write_only: true` 只能与 `mutated: true` 同时使用；
- `alias` 指向的必须是一个会被写入的输入；
- 声明了 `alias` 的输出不能再声明 `buffer`；
- 至多一个输出声明 `buffer: out`；
- `overlaps` 只能写在被写入的输入或 `buffer: out` 输出上，列出的是其他输入，每个只列一次。

validator 对每个 effect 分支检查 operator schema、别名关系与 roofline 的读写计数是否一致。

## 7. metadata 张量：generator 与 requires {#generators}

varlen、paged 等 op 使用 metadata 张量，例如 `cu_seqlens`、`block_table`。metadata 张量的类型写在签名中，取值则由 generator 在实例化时生成，generator 写在张量的 `values` 字段中。以 MeanPooling 为例：

```yaml
forall: {B: Dim, S: Dim, H: Dim, D: Dim, NS: Dim, NC: Dim, T: "DType[...]", seq_lens: "Seq[Int]"}
inputs:
  cu_seqlens: {dtype: int32, type: "Tensor | None", default: null, shape: "[NS + 1]",
               values: "prefix_sum(seq_lens)", requires: ["prefix_offsets(S)"]}
  chunk_indices: {dtype: int32, type: "Tensor | None", default: null, shape: "[NC, 2]",
                  values: "chunk_indices(seq_lens, chunk_size)"}
shape_rules:
- "present(chunk_indices) == present(cu_seqlens)"
```

generator 的规则如下：

- generator 的实参由 workload 行给出，例如长度列表 `seq_lens`。`forall` 中 kind 为 `Seq[Int]` 的名字只能用作 generator 的实参。workload 行中的 `Seq[Int]` 可以写成整数列表，也可以写成返回列表的值 primitive 调用，例如 `seq_lens: "repeat(512, 64)"`。
- 实例化时，generator 的结果与声明的形状做 unification，形状中的其他 index 由此求得，不需要在行中给出。上例中的 `NS` 与 `NC` 分别由 `cu_seqlens` 与 `chunk_indices` 的生成结果求得。实际调用时，这些 index 同样由输入张量求得。
- generator 或者是确定性的，或者使用由 workload 种子派生的私有随机数，因此同一条行每次生成的取值都相同。
- 每个 generator 结果的秩是固定的，或者由其形状参数决定。
- 生成的张量声明整数 dtype（`int32` 或 `int64`）；参数超出定义域，或结果超出所声明 dtype 的范围时，generator 会报错。
- generator 的实参可以是返回列表的 primitive，例如 GroupedGEMM 的 `as_tensor(balanced_sizes(M, G))`。

`requires` 的规则如下：

- `requires` 列出约束 metadata 张量内容的谓词。例如 `prefix_offsets(S)` 要求张量首项为 0、单调不减、末项为 `S`。被约束张量的内容是谓词隐含的第一个实参。
- 每个谓词按固定的秩读取被约束的张量；逐元素给出上下界的谓词（如 `in_range`）可以作用于任意秩的张量。
- 谓词的实参可以是另一个 metadata 张量，这样一条谓词就能约束两个张量之间的关系，例如 GroupedGEMM 的 `batch_offsets` 声明 `requires: ["exclusive_prefix_of(batch_sizes)"]`，要求它是 `batch_sizes` 的 exclusive 前缀和。在被约束的张量存在的每个分支上，作为实参的那个张量也必须存在。
- validator 在实例化时用生成的取值检查 `requires`。在实际调用中，这些约束由调用方保证，op 默认不检查张量的内容；因此 validator 还会检查，在被约束的张量存在的每个分支上，谓词都是良定义的。
- 声明了 `requires` 的张量也必须声明 `values`。
- 有输入的 `requires` 含 `prefix_offsets` 的 op，在 `params` 中声明 `validate_inputs: {type: bool, default: false, kw_only: true}`。输入的 `requires` 含 `in_range`、不含 `prefix_offsets` 的 op 也可以声明它。
- 调用方设置 `validate_inputs` 时，由 manifest 生成的检查在实现运行前于 CPU 上读取输入，检查其中的 `prefix_offsets` 与 `in_range` 谓词。其他谓词仍由调用方保证。
- 这项检查会同步设备，不能在 CUDA Graph 捕获中运行。

## 8. dtype 组合与打包 dtype {#dtype-combos}

当多个 `DType` index 只允许特定组合时，spec 用 `dtype_combos` 列出所有允许的组合。以 paged GQA 为例：

```yaml
forall: {..., T: "DType[float16 | bfloat16 | float8_e4m3fn]", KV: "DType[float16 | bfloat16 | float8_e4m3fn]"}
dtype_combos:
- {T: float16, KV: float16}
- {T: bfloat16, KV: bfloat16}
- {T: float16, KV: float8_e4m3fn}
- {T: bfloat16, KV: float8_e4m3fn}
- {T: float8_e4m3fn, KV: float8_e4m3fn}
```

- 每一行是一个从 index 到 dtype 的映射。各行的键相同，键中可以包含 dtype 参数；各行互不相同。
- 一次调用的 dtype 取值必须与其中某一行完全相同。
- `dtype_combos` 的每一列在 spec 接受的每个分支上都必须被用到。

fp4、int4 等打包 dtype 存放在 `uint8` 等载体 dtype 中，spec 按载体书写：

- `dtype` 写载体 dtype，`shape` 写 PyTorch 所见的载体形状，例如 GemmW4A16 的 `packed_weight: "[N, K // 2]"`；
- 逻辑 dtype 由 dtype 参数给出，或在 spec 中固定；
- roofline 按载体计算字节数。

## 9. 取值范围随 dtype 变化的标量参数 {#scalar-dtype}

有些标量参数的合法取值取决于调用时的 dtype，例如 Add 的 `alpha`。这类约束使用 primitive `category` 与 `representable` 写成 refinement：

```yaml
params:
  alpha: {type: int | float, default: 1, kw_only: true}
shape_rules:
- "category(alpha) == 'int' or category(alpha) == category(T)"
- "representable(alpha, T)"
```

生成的调用检查对 TileOPs in-tree 实现与 target 提供的实现应用同一条规则。

## 10. 组合 op：composition {#composition}

组合 op 用 `composition` 记录其 in-tree 实现可能持有的子 op，以及它自身 kernel 的位置。以 FusedMoESharedExpert 为例：

```yaml
composition:
  kind: composite
  stages:
  - {name: route_select, op: FusedTopKFwdOp}
  - {name: routed_experts, op: FusedMoEExpertsFwdOp}
  - {name: shared_expert, op: SharedExpertMLPFwdOp, optional: true}
```

- 每个 stage 或者引用一个 manifest 中的 op（`op`），或者引用 op 自身 `kernel_types` 中的一个键（`kernel`）。
- 并非每次调用都会持有的子 op，写成 `optional: true` 的 stage；是否持有由代码决定。
- 子 op 的构造时机与次数、调度方式以及 forward 的执行，都由代码决定。
- manifest 不规定组合 op 的 roofline 与其各 stage 的 roofline 之间的关系。
- 对于 `implemented` 的 op，validator 按顺序核对：`op` stage 与类的 `delegate_types` 一致，`kernel` stage 与类的 `kernel_types` 一致。
- `stages` 不能为空，stage 名互不相同，`optional` 为布尔值。

## 11. 张量序列 {#sequences}

持有张量列表的实参声明 `type: "list[Tensor]"`。它的逐元素 index 的 kind 是 `Seq[Dim]`、`Seq[Shape]` 与 `Seq[DType[...]]`。以 Concat 为例：

```yaml
forall: {S: "Seq[Shape]", D: "Seq[DType[float16 | bfloat16 | float32]]"}
params:
  dim: {type: int, default: 0}
inputs:
  tensors: {type: "list[Tensor]", dtype: D, shape: "[*S]"}
outputs:
  output: {dtype: "promote(D)", shape: "[*concat(S, dim)]"}
shape_rules:
- "len(S) >= 1"
workloads:
- {S: [[2048, 1024], [2048, 4096], [2048, 3000]], dim: 1, dtype_cases: [{D: [bfloat16, bfloat16, bfloat16]}], label: uneven-cols}
```

- 在 `list[Tensor]` 声明的 `shape` 与 `dtype` 中，每个逐元素 index 都表示该元素对应的那一项：第 `i` 个元素的形状是 `S[i]`，dtype 是 `D[i]`，那里的 `len(S)` 是 `len(S[i])`。其他 index 由所有元素共用。
- 在该声明之外，逐元素 index 表示整个列表，可以在输出、refinement、`let` 与 roofline 公式中读取。`let` 不展开到元素上。
- `Seq[Dim]` 为每个元素保存一个轴长；`Shape` 则是所有元素共用的一个轴元组。
- 序列长度是它的逐元素 index 的共同长度，`list[Tensor]` 声明至少读取一个逐元素 index。
- 逐元素 index 在每个元素上求解。共用的 index 先从其他张量求解；只有推断无法继续时，才从第 0 个元素求解，再在每个元素上检查。
- 空序列使它的逐元素 index 取空列表；只有该序列能求出的共用 index 保持未求得，调用报错，并指出对应的声明。
- 张量序列每次调用都要传入，并且只被读取。它不声明 effect、`values`、`requires` 或 `device`，也不作为并集的成员。
- 确定调用设备、16 字节对齐的副本、别名与重叠检查都逐个元素进行。推导字节数时每个元素计一次，`bytes(t)` 是各元素字节数之和。
- workload 行把逐元素 index 写成每个元素一项的列表。`Seq[DType]` index 在 `dtype_cases` 中写成 dtype 列表，它在 case id 中的值以 `.` 连接，例如 `uneven-cols-bfloat16.bfloat16.bfloat16`。
- `list[Tensor]` 输出、序列中的 `None` 元素、作为并集成员的 `list[Tensor]`，以及序列上的 effect，都不被接受。
