# 扩展写法

本页介绍只有部分 spec 才会用到的写法。每一节对应一种情况，各节可以独立阅读。大多数 spec 所需的字段见[写一个 spec](writing.md)。

## 1. 可选输入、可空输出与可缺省参数 {#presence}

张量是否存在通过 `present` 表达。以 FusedMoESharedExpert 为例：

```yaml
inputs:
  correction_bias:  {dtype: float32, shape: "[E]", optional: true}
  shared_w_gate_up: {dtype: D, shape: "[2 * S, H]", optional: true}
  shared_w_down:    {dtype: D, shape: "[H, S]", optional: "present(shared_w_gate_up)"}
outputs:
  shared_output: {dtype: D, shape: "[T, H]", nullable: "present(shared_w_gate_up)"}
  routed_output: {dtype: D, shape: "[T, H]"}
```

- 可选输入使用 `optional: true` 声明。如果几个张量必须同时传入或同时不传，它们共用同一个 discriminant，写作 `optional: "<表达式>"`，例如上例中的 `shared_w_down`。
- 可能返回 `None` 的输出使用 `nullable: "<表达式>"` 声明。在上例中，传入 `shared_w_gate_up` 时 op 返回 `shared_output`，否则该输出为 `None`。
- `optional` 与 `nullable` 的表达式只能由取值有限的布尔量组成，包括 `Bool` 与枚举参数、ADT 的 tag 与取值有限的字段，以及 `present(...)`。
- 可选输入排在必选输入之后，`forward` 按声明顺序接收它们，默认值为 `None`。省略一个可选输入与显式传入 `None` 效果相同。
- `int | None` 参数的 kind 是 `Maybe[Int]`：`present(v)` 表示它是否给出，`v.value` 是它的值。`v.value` 只能出现在 `present(v)` 为 true 的分支中。
- 一个 index 只在出现它的分支上被用到。在上例中，不传 `shared_w_gate_up` 时 `S` 没有被用到，对应的 workload 行不需要给出 `S`。
- workload 行通过 `some` 列出传入的可选张量。`some` 只列出声明为 `optional: true` 的张量；`optional` 为表达式的张量随表达式的取值确定，不写在 `some` 中。例如传入共享专家时写 `some: [shared_w_gate_up]`，`shared_w_down` 随之传入。
- 对于 `implemented` 的 op，每个可选张量都要至少有一条行传入它，并至少有一条行不传它。

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
- `match` 的对象必须是取值有限的 discriminant，即 `Bool`、枚举、ADT、`present(...)`，或由它们组成的元组。对元组做 match 时，`when` 写成列表，例如 `when: [true, false]`。
- `cases` 在 spec 接受的所有取值上必须既无遗漏也无重叠。
- 多个张量应用同一个 type family 时，它们总是取同一个分支。
- type family 之间的引用不能成环；如果某个 type family 没有被任何形状使用，validator 会报错。

只读取 discriminant 的 refinement 称为**定义域限制**。定义域限制在选择 type family 分支之前检查，被它排除的取值不需要对应的 case。以 Clamp 为例：

```yaml
types:
  ClampOut:
    params: {A: Shape, L: Shape, U: Shape, pl: Bool, pu: Bool}
    match: [pl, pu]
    cases:
    - {when: [true, true],  is: "[*broadcast(A, L, U)]"}
    - {when: [true, false], is: "[*broadcast(A, L)]"}
    - {when: [false, true], is: "[*broadcast(A, U)]"}
outputs:
  output: {dtype: T, shape: "ClampOut[A, L, U, present(min), present(max)]"}
shape_rules:
- "present(min) or present(max)"
```

`present(min) or present(max)` 只读取 discriminant，它排除了 `min` 与 `max` 都不传的情况，因此三个 case 已经覆盖了 Clamp 接受的所有组合。判断一条 refinement 是否只读取 discriminant 时，考察的是它出现的所有名字，与操作数的书写顺序无关。

## 3. 带字段的参数：ADT {#adt}

当一个参数有有限的几种形态，且每种形态带有各自的字段时，参数的类型使用 ADT 描述。ADT 定义在 `spec/types.yaml` 中，供多个 spec 共用：

```yaml
adts:
  MGroupedLayout:
    sum:
      contiguous:
        python: tileops.ops.moe.contracts.ContiguousLayoutSpec
        fields:
          packing: {type: "'tight' | 'aligned'", python: tileops.ops.moe.contracts.ContiguousPacking}
          metadata_kind: {type: "'physical_psum' | 'per_row'", python: tileops.ops.moe.contracts.ContiguousMetadata}
          alignment: Dim
        invariant: "(packing == 'tight') == (alignment == 1) and alignment >= 1"
      masked:
        python: tileops.ops.moe.contracts.MaskedLayoutSpec
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

- `let` 的值由签名计算：能在构造时求值的在构造时计算，其余的在每次调用时计算。workload 行中不需要给出 `let`。
- 一个 `let` 可以引用其他 `let`，例如 `sH` 引用了 `kH`，但 `let` 之间的依赖不能成环。

primitive 是表达式中可以调用的内建函数，例如 `broadcast`、`reduced`、`per_axis`、`ceil_div`。

- primitive、generator 与 `requires` 中的谓词都是固定的集合，完整列表在 `tileops.manifest.primitives` 中，每个成员的 docstring 说明了它的计算内容。
- 只供某一个 family 使用的成员带有 family 前缀，例如 `pool.out`、`moe.capacity`。
- 参数超出定义域时，primitive 会报错，错误信息指向调用它的那条声明。
- 新增成员需要修改 `tileops.manifest.primitives`，并附带测试。
- 所有接受轴参数的 primitive 都按同一规则处理轴：对零维张量，`0` 与 `-1` 都表示唯一的标量轴；其他情况下，轴的取值范围是 `[-rank, rank)`。

## 5. 构造期张量、内存布局与设备 {#placement}

- 在构造时传入的张量声明在 `params` 中，带有 `dtype` 与 `shape`，并可以声明 `optional: true`。例如 LongRoPE 的 `rescale_factors`：

  ```yaml
  params:
    rescale_factors: {dtype: R, shape: "[D // 2]", optional: true}
  ```

- 要求内存连续的张量声明 `contiguous: true`，例如 MoE staged 系列 op 的输入与输出。没有这项声明的张量可以有任意 stride。
- 必须位于 CPU 上的张量声明 `device: cpu`，例如 GatedDeltaNet 的 `cu_seqlens_cpu`。
- 没有调用期张量输入的 op 声明 `device` 参数，例如 Alibi。调用设备的确定方式见[调用与校验 4](calls.md#device)。

## 6. 对参数的写入：effect {#effects}

没有 effect 声明的 op 只读取输入，并为输出分配新的张量。如果 op 会写入参数，需要在签名的相应张量上声明 effect。effect 决定了 roofline 的读写计数，以及声明了 compile boundary 的 op 所生成的 operator schema。

**表 1** effect 声明

| No. | 声明 | 含义 | 例子 |
| --- | --- | --- | --- |
| 1 | 输出上的 `buffer: out` | `forward` 在所有输入之后增加参数 `out`。调用方传入 `out` 时，op 将结果写入并返回它；未传入时，op 分配新张量。`out` 与该输出的形状和 dtype 相同 | MoEGroupedGemm 的 `output` |
| 2 | 输入上的 `mutated: true` | op 可能写入这个输入，它在调用前的内容参与计算 | |
| 3 | 输入上的 `mutated: true` 与 `write_only: true` | 必须传入的结果缓冲：op 覆盖写入，结果只取决于其他输入；如果 op 返回 `None`，`outputs` 为空 | FusedMoEExperts 的 `output` |
| 4 | 输入上的 `mutated: "<discriminant 表达式>"` | 仅当表达式为 true 时，op 才写入这个输入 | 激活函数的 `mutated: inplace` |
| 5 | 输出上的 `alias: <输入名>` | 该输入被写入时，这个输出就是该输入对象本身 | 激活函数的 `alias: input` |

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

- `write_only: true` 只能与 `mutated: true` 同时使用；
- `alias` 指向的必须是一个会被写入的输入；
- 声明了 `alias` 的输出不能再声明 `buffer`；
- 至多一个输出声明 `buffer: out`。

validator 对每个 effect 分支检查 operator schema、别名关系与 roofline 的读写计数是否一致。

## 7. metadata 张量：generator 与 requires {#generators}

varlen、paged 等 op 使用 metadata 张量，例如 `cu_seqlens`、`block_table`。metadata 张量的类型写在签名中，取值则由 generator 在实例化时生成，generator 写在张量的 `values` 字段中。以 MeanPooling 为例：

```yaml
forall: {B: Dim, S: Dim, H: Dim, D: Dim, NS: Dim, NC: Dim, T: "DType[...]", seq_lens: "Seq[Int]"}
inputs:
  offsets: {dtype: int32, optional: true, shape: "[NS + 1]",
            values: "prefix_sum(seq_lens)", requires: ["prefix_offsets(S)"]}
  indices: {dtype: int32, optional: "present(offsets)", shape: "[NC, 2]",
            values: "chunk_indices(seq_lens, chunk_size)"}
```

generator 的规则如下：

- generator 的实参由 workload 行给出，例如长度列表 `seq_lens`。`forall` 中 kind 为 `Seq[Int]` 的名字只能用作 generator 的实参。workload 行中的 `Seq[Int]` 可以写成整数列表，也可以写成返回列表的值 primitive 调用，例如 `seq_lens: "repeat(512, 64)"`。
- 实例化时，generator 的结果与声明的形状做 unification，形状中的其他 index 由此求得，不需要在行中给出。上例中的 `NS` 与 `NC` 分别由 `offsets` 与 `indices` 的生成结果求得。实际调用时，这些 index 同样由输入张量求得。
- generator 或者是确定性的，或者使用由 workload 种子派生的私有随机数，因此同一条行每次生成的取值都相同。
- 每个 generator 结果的秩是固定的，或者由其形状参数决定。
- 生成的张量声明整数 dtype（`int32` 或 `int64`）；参数超出定义域，或结果超出所声明 dtype 的范围时，generator 会报错。
- generator 的实参可以是返回列表的 primitive，例如 GroupedGemm 的 `as_tensor(balanced_sizes(M, G))`。

`requires` 的规则如下：

- `requires` 列出约束 metadata 张量内容的谓词。例如 `prefix_offsets(S)` 要求张量首项为 0、单调不减、末项为 `S`。被约束张量的内容是谓词隐含的第一个实参。
- 每个谓词按固定的秩读取被约束的张量；逐元素给出上下界的谓词（如 `in_range`）可以作用于任意秩的张量。
- 谓词的实参可以是另一个 metadata 张量，这样一条谓词就能约束两个张量之间的关系，例如 GroupedGemm 的 `batch_offsets` 声明 `requires: ["exclusive_prefix_of(batch_sizes)"]`，要求它是 `batch_sizes` 的 exclusive 前缀和。在被约束的张量存在的每个分支上，作为实参的那个张量也必须存在。
- validator 在实例化时用生成的取值检查 `requires`。在实际调用中，这些约束由调用方保证，op 不检查张量的内容；因此 validator 还会检查，在被约束的张量存在的每个分支上，谓词都是良定义的。
- 声明了 `requires` 的张量也必须声明 `values`。

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

## 10. 复合 op：composition {#composition}

复合 op 用 `composition` 记录其 in-tree 实现可能持有的子 op，以及它自身 kernel 的位置。以 FusedMoESharedExpert 为例：

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
- manifest 不规定复合 op 的 roofline 与其各 stage 的 roofline 之间的关系。
- 对于 `implemented` 的 op，validator 按顺序核对：`op` stage 与类的 `delegate_types` 一致，`kernel` stage 与类的 `kernel_types` 一致。
- `stages` 不能为空，stage 名互不相同，`optional` 为布尔值。
