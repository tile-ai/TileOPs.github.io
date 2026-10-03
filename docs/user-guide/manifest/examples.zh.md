# 示例

本页收录 manifest 中的五个真实 spec，每个 spec 之后附有一张表，列出其中各处写法对应的说明。这些 spec 摘自 `src/tileops/manifest/`，删去了注释与部分 workload 行。

## 1. GemmFwdOp：type family {#gemm}

```yaml
# src/tileops/manifest/spec/gemm.yaml
GemmFwdOp:
  ref_api: "torch.matmul"
  family: gemm
  status: implemented

  signature:
    types:
      Mat:
        params: {t: Bool, R: Dim, C: Dim}
        match: t
        cases:
        - {when: false, is: "[R, C]"}
        - {when: true, is: "[C, R]"}
    forall: {M: Dim, N: Dim, K: Dim, T: "DType[float16 | bfloat16]"}
    params:
      trans_a: {type: bool, default: false}
      trans_b: {type: bool, default: true}
    inputs:
      a: {dtype: T, shape: "Mat[trans_a, M, K]"}
      b: {dtype: T, shape: "Mat[trans_b, K, N]"}
    outputs:
      d: {dtype: T, shape: "[M, N]"}

  workloads:
  - {M: 1024, N: 1024, K: 1024, trans_a: false, trans_b: false, dtype_cases: [{T: float16}, {T: bfloat16}], label: square-1k}
  - {M: 128, N: 2112, K: 7168, trans_a: false, trans_b: true, dtype_cases: [{T: bfloat16}], label: ds-v3-decode-qkv-a}
  # 其余行省略

  roofline:
    flops: "2 * M * N * K"
```

**表 1** `GemmFwdOp` 中各处写法的说明

| No. | spec 中的写法 | 说明所在 |
| --- | --- | --- |
| 1 | `types.Mat`：由 `trans_a`、`trans_b` 决定 `a`、`b` 的轴顺序 | [扩展写法 2](extensions.md#type-family) |
| 2 | `forall` 声明三个轴长与一个 `DType` index | [写一个 spec 3](writing.md#forall) |
| 3 | `trans_a`、`trans_b` 既是构造参数，也是 `Mat` 的 discriminant | [写一个 spec 4](writing.md#params) |
| 4 | workload 行给出参数与 index 的取值；第一行展开为两个 case，id 分别是 `square-1k-float16` 与 `square-1k-bfloat16` | [写一个 spec 8](writing.md#workloads) |
| 5 | `roofline` 只给出 `flops`；`bytes` 由签名推导，等于 `(M*K + K*N + M*N)` 乘以每个元素的字节数 | [写一个 spec 9](writing.md#roofline) |

## 2. MoEPrePermuteFwdOp：ADT、let 与 generator {#moe}

`layout` 参数的类型是 `spec/types.yaml` 中定义的 ADT `MGroupedLayout`，定义见[扩展写法 3](extensions.md#adt)。

```yaml
# src/tileops/manifest/spec/moe.yaml
MoEPrePermuteFwdOp:
  family: moe
  status: implemented
  signature:
    types:
      LayoutMetadata:
        params: {layout: MGroupedLayout, E: Dim, P: Dim}
        match: layout
        cases:
        - {when: {contiguous: {metadata_kind: physical_psum}}, is: "[E]"}
        - {when: {contiguous: {metadata_kind: per_row}}, is: "[P]"}
    forall: {T: Dim, H: Dim, K: Dim, D: "DType[float16 | bfloat16]"}
    params:
      layout: {type: MGroupedLayout}
      num_local_experts: {type: int}
    inputs:
      hidden_states: {dtype: D, shape: "[T, H]", contiguous: true}
      local_expert_ids: {dtype: int32, shape: "[T, K]", contiguous: true, values: "topk_ids(T, K, num_local_experts)", requires: ["in_range(0, num_local_experts)"]}
    let:
      P: "moe.capacity(layout, T * K, num_local_experts)"
    outputs:
      expert_input: {dtype: D, shape: "[P, H]", contiguous: true}
      layout_metadata: {dtype: int32, shape: "LayoutMetadata[layout, num_local_experts, P]", contiguous: true}
      inverse_indices: {dtype: int32, shape: "[T * K]", contiguous: true}
    shape_rules:
    - "layout.kind == 'contiguous'"
    - "num_local_experts > 0"
    - "T > 0"
    - "K > 0"
  workloads:
  - {T: 32, H: 128, K: 2, num_local_experts: 4, layout: {contiguous: {packing: tight, metadata_kind: physical_psum, alignment: 1}}, dtype_cases: [{D: float16}, {D: bfloat16}], label: "decode-t32"}
  # 其余行省略
  roofline:
    flops: "0"
```

**表 2** `MoEPrePermuteFwdOp` 中各处写法的说明

| No. | spec 中的写法 | 说明所在 |
| --- | --- | --- |
| 1 | `layout.kind == 'contiguous'` 是一条定义域限制，它排除了 masked layout，因此 `LayoutMetadata` 不需要 masked 对应的 case | [扩展写法 2](extensions.md#type-family)、[3](extensions.md#adt) |
| 2 | `LayoutMetadata` 按 constructor 与字段 `metadata_kind` 做 pattern matching | [扩展写法 2](extensions.md#type-family) |
| 3 | `int` 参数 `num_local_experts` 直接出现在形状中 | [写一个 spec 4](writing.md#params) |
| 4 | `let` 中的 `P` 由 primitive `moe.capacity` 计算得到，并用于输出形状 | [扩展写法 4](extensions.md#let) |
| 5 | `local_expert_ids` 的取值由 generator `topk_ids` 生成，其内容由 `requires` 约束 | [扩展写法 7](extensions.md#generators) |
| 6 | 各张量声明了 `contiguous: true` | [扩展写法 5](extensions.md#placement) |
| 7 | workload 行中的 `layout` 写成 ADT 值 | [扩展写法 3](extensions.md#adt) |

## 3. MeanPoolingFwdOp：可选输入与 metadata 张量 {#meanpool}

```yaml
# src/tileops/manifest/spec/pool.yaml
MeanPoolingFwdOp:
  family: pool
  status: implemented

  signature:
    types:
      PoolOut:
        params: {p: Bool, B: Dim, NC: Dim, S: Dim, c: Dim, H: Dim, D: Dim}
        match: p
        cases:
        - {when: true, is: "[B, NC, H, D]"}
        - {when: false, is: "[B, ceil_div(S, c), H, D]"}
    forall: {B: Dim, S: Dim, H: Dim, D: Dim, NS: Dim, NC: Dim, T: "DType[float16 | bfloat16 | float32]", seq_lens: "Seq[Int]"}
    params:
      chunk_size: {type: int}
      accum_dtype: {type: torch.dtype}
    inputs:
      x: {dtype: T, shape: "[B, S, H, D]"}
      offsets: {dtype: int32, optional: true, shape: "[NS + 1]", values: "prefix_sum(seq_lens)", requires: ["prefix_offsets(S)"]}
      indices: {dtype: int32, optional: "present(offsets)", shape: "[NC, 2]", values: "chunk_indices(seq_lens, chunk_size)"}
    outputs:
      output: {dtype: T, shape: "PoolOut[present(offsets), B, NC, S, chunk_size, H, D]"}
    shape_rules:
    - "chunk_size > 0 and chunk_size % 32 == 0"
    - "D <= 128 or D % 128 == 0"
    - "category(accum_dtype) == 'float'"

  workloads:
  - {B: 1, S: 8192, H: 64, D: 128, chunk_size: 64, accum_dtype: float32, dtype_cases: [{T: float16}, {T: bfloat16}], label: uniform-8k}
  - {B: 1, S: 8192, H: 64, D: 128, seq_lens: "repeat(2048, 4)", chunk_size: 64, accum_dtype: float32, some: [offsets], dtype_cases: [{T: float16}, {T: float32}], label: ragged-even}
  # 其余行省略

  roofline:
    flops: "B * S * H * D + B * (NC if present(offsets) else ceil_div(S, chunk_size)) * H * D"
```

**表 3** `MeanPoolingFwdOp` 中各处写法的说明

| No. | spec 中的写法 | 说明所在 |
| --- | --- | --- |
| 1 | `indices` 声明为 `optional: "present(offsets)"`，因此与 `offsets` 同时传入或同时不传 | [扩展写法 1](extensions.md#presence) |
| 2 | type family `PoolOut` 以 `present(offsets)` 为 discriminant | [扩展写法 2](extensions.md#type-family) |
| 3 | 第一行不传 `offsets`，没有用到 `seq_lens`，因此行中不给出它；第二行通过 `some: [offsets]` 传入 `offsets`，同时给出 `seq_lens`，这里写成值 primitive 调用 `"repeat(2048, 4)"` | [扩展写法 1](extensions.md#presence)、[写一个 spec 8](writing.md#workloads) |
| 4 | `NS` 与 `NC` 由 generator 结果的 unification 求得，不在行中给出 | [扩展写法 7](extensions.md#generators) |
| 5 | `category(accum_dtype) == 'float'` 限制累加 dtype 必须是浮点类型 | [扩展写法 9](extensions.md#scalar-dtype) |
| 6 | roofline 公式通过 `present(offsets)` 区分两种情况 | [写一个 spec 9](writing.md#roofline) |

## 4. ReluFwdOp：任意秩与 effect {#relu}

```yaml
# src/tileops/manifest/spec/elementwise_unary_activation.yaml
ReluFwdOp:
  ref_api: "torch.nn.functional.relu"
  family: elementwise
  status: implemented

  signature:
    forall: {S: Shape, T: "DType[float16 | bfloat16 | float32]"}
    params:
      inplace: {type: bool, default: false, kw_only: true}
    inputs:
      input: {dtype: T, shape: "[*S]", mutated: inplace}
    outputs:
      output: {dtype: T, shape: "[*S]", alias: input}

  workloads:
  - {S: [2048, 4096], dtype_cases: [{T: float16}, {T: bfloat16}], label: "hidden-state-prefill"}
  - {S: [1, 4096], dtype_cases: [{T: bfloat16}], label: "hidden-state-decode"}

  roofline:
    flops: "prod(S)"
```

**表 4** `ReluFwdOp` 中各处写法的说明

| No. | spec 中的写法 | 说明所在 |
| --- | --- | --- |
| 1 | `S` 的 kind 是 `Shape`，输入与输出使用同一个形状项 `[*S]` | [写一个 spec 3](writing.md#forall)、[5](writing.md#tensors) |
| 2 | `mutated: inplace` 与 `alias: input` 表示 `inplace` 为 true 时 op 写入输入并将其返回 | [扩展写法 6](extensions.md#effects) |
| 3 | workload 行中的 `S` 写成整数列表 | [写一个 spec 8](writing.md#workloads) |

## 5. AlibiFwdOp：没有张量输入 {#alibi}

```yaml
# src/tileops/manifest/spec/elementwise_generative.yaml
AlibiFwdOp:
  family: elementwise
  status: implemented

  signature:
    params:
      seq_len: {type: int, kw_only: true}
      num_heads: {type: int, kw_only: true}
      out_dtype: {type: "float16 | bfloat16 | float32", default: float32, kw_only: true}
      device: {type: "torch.device | str | None", default: null, kw_only: true}
    outputs:
      output: {dtype: out_dtype, shape: "[num_heads, seq_len, seq_len]"}
    shape_rules:
    - "seq_len > 0"
    - "num_heads > 0"

  workloads:
  - {seq_len: 2048, num_heads: 32, out_dtype: float16, label: "mpt-7b-2k"}
  - {seq_len: 2048, num_heads: 32, out_dtype: bfloat16, label: "mpt-7b-2k"}
  # 其余行省略

  roofline:
    flops: "3 * num_heads * seq_len * seq_len"
```

**表 5** `AlibiFwdOp` 中各处写法的说明

| No. | spec 中的写法 | 说明所在 |
| --- | --- | --- |
| 1 | 所有 index 都来自构造参数，因此 spec 中没有 `forall` | [写一个 spec 4](writing.md#params) |
| 2 | 输出的 `dtype` 直接使用 dtype 参数 `out_dtype` | [写一个 spec 7](writing.md#dtype) |
| 3 | op 没有调用期张量输入，因此声明了 `device` 参数 | [扩展写法 5](extensions.md#placement)、[调用与校验 4](calls.md#device) |
| 4 | 两行的 `label` 相同，case id 以 dtype 参数的值区分，分别是 `mpt-7b-2k-float16`、`mpt-7b-2k-bfloat16` | [写一个 spec 8](writing.md#workloads) |
