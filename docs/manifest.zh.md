# 读写 manifest

传统算子库以实现为中心：算子逐个写、逐个调优，支持哪些形状、哪些 dtype、跑多快，都由实现事后说明，文档写的是追述。

TileOPs 的组织方式相反：算子的规格先声明，实现由规格推导。每个算子的规格称为它的 **spec**，是 [`src/tileops/manifest/spec/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/manifest/spec) 下以 family 命名的 YAML 文件里的一个条目；这些文件合起来就是 manifest。

**一个算子有了 spec，就成为整个系统的数据输入。** 各个环节读同一份声明，而不是各自去读实现：

| 谁消费 | 读 spec 里的什么 | 产出 |
| --- | --- | --- |
| 算子层 | `signature` | 每次调用前后的检查、输出形状推导、dtype 检查，以及 `torch.compile` 看到的 operator |
| [契约测试](https://github.com/tile-ai/TileOPs/tree/main/tests) | `workloads` | 每个 workload 行的每个 dtype case 对应一次调用，交给算子执行 |
| [每晚的 benchmark](https://github.com/tile-ai/TileOPs/tree/main/benchmarks) | `workloads` | 这些调用各自的 device time |
| [roofline](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/perf) | `roofline` | 一次调用的计算量与访存量，效率的分母 |
| 本文档站 | `signature`、`workloads` | Benchmarks 页每一行下面列出的形状 |
| CI 的 [spec 校验器](https://github.com/tile-ai/TileOPs/blob/main/scripts/validate_manifest.py) | 全部字段 | 检查声明与实现是否一致，见 [spec 校验器](#spec-validator) |

**每一行都以 spec 为前提**：没有 spec，就没有生成的校验、没有契约测试、没有性能数据，CI 也拦不下任何回退。**为一个算子写 manifest 不是补文档，而是把它接进这条数据流。**{ .keystone }

本页依次讲：一个条目的构成、怎么读、怎么分五步写、manifest 里反复出现的四种写法，以及校验器查什么。完整规则见 [Op Manifest](design/manifest.md)，本页不重复。

## 一个条目的构成

每个 family 文件是一个 `算子名 → 条目` 的映射；大的 family 拆成若干个 `<family>_<shard>.yaml`。加载时各文件合并，算子名重复即报错。多个条目共用的代数数据类型写在 `spec/types.yaml`。

条目的键是算子的 Python 类名 `{Name}[{Fwd|Bwd}]Op`，另一个方向也有条目时方向后缀必须写，校验器要求 `cls.__name__` 与它逐字相同。

| 字段 | 必填 | 内容 |
| --- | --- | --- |
| `family` | 是 | 公开模块：算子从 `tileops.<family>.<Op>` 导入 |
| `status` | 是 | `implemented`；还没有符合 spec 的实现时写 `spec-only` |
| `ref_api` | 否 | 算子语义所依照的 API 的全限定名，如 `torch.nn.functional.rms_norm` |
| `signature` | 是 | 算子的类型，见下表 |
| `workloads` | 是 | 测试与 benchmark 执行的调用 |
| `roofline` | 是 | 一次调用的开销，规范见 [Roofline](design/roofline.md) |
| `composition` | 否 | 复合算子按顺序列出的各阶段：可能持有的子算子类，以及它自己的 kernel key |

签名是一个以具名类型变量为参数的函数类型：

| 子字段 | 内容 |
| --- | --- |
| `forall` | 每个自由类型变量及其种类：`Dim`（一个轴的长度）、`Shape`（若干个轴）、`DType[...]`（一组 dtype 中的一个）、`Seq[Int]`（只作生成器参数的整数列表） |
| `params` | `__init__` 的参数：`type`，可选的 `default` 与 `kw_only` |
| `inputs` / `outputs` | 张量，每个写成 `{dtype, shape}`，用上面的类型变量表达，另可带是否可选与副作用的标记 |
| `types` | 类型族：按某个开关的取值选定形状 |
| `let` | 由类型变量算出的量 |
| `shape_rules` | 约束：关于类型变量取值的谓词 |
| `dtype_combos` | 多个 `DType` 变量不是任意组合都支持时，列出支持的组合 |

键的顺序就是位置：`params` 对应 `__init__`，`inputs` 对应 `forward`，`outputs` 对应返回的 tuple，调换顺序是不兼容的改动。

## 读一份 spec

`RMSNormFwdOp`，八个 workload 行里取两行：

```yaml
RMSNormFwdOp:
  ref_api: torch.nn.functional.rms_norm
  family: norm
  status: implemented
  signature:
    forall: {B: Shape, T: "DType[float16 | bfloat16]"}
    params:
      normalized_shape: {type: "list[int] | tuple[int, ...]"}
      eps: {type: "float | None", default: null}
    inputs:
      x: {dtype: T, shape: "[*B, *normalized_shape]"}
      weight: {dtype: T, shape: "[*normalized_shape]", optional: true}
    outputs:
      output: {dtype: T, shape: "[*B, *normalized_shape]"}
    shape_rules:
      - "len(normalized_shape) > 0"
  workloads:
    - {B: [2048], normalized_shape: [4096], eps: 1.0e-06, some: [weight],
       dtype_cases: [{T: float16}, {T: bfloat16}], label: llama-8b-prefill}
    - {B: [4, 2048], normalized_shape: [128], dtype_cases: [{T: bfloat16}], label: qk-norm-head}
  roofline:
    flops: "(4 if present(weight) else 3) * prod(B) * prod(normalized_shape)"
```

分五步读：

1. **`forall`**：调用之间哪些量会变。`B` 是任意个数的前导轴，`T` 是 dtype。
2. **`inputs` 与 `outputs`**：同名即相等。`x`、`weight`、`output` 的 dtype 都是 `T`，三者一致；`x` 与 `output` 写的是同一个形状，所以形状相同。`weight` 标了 `optional: true`，调用时可以不传。
3. **`params`**：`normalized_shape` 是构造参数，同时用 `*` 展开进形状里：`x` 的末尾几个轴必须等于它。
4. **`shape_rules`**：形状表达不了的约束。这里是至少归一化一个轴。
5. **`workloads`**：每一行配上它的每个 `dtype_cases` 就是一次调用。一行给出类型变量（`B`）、构造参数，并在 `some` 里列出这次传入的可选张量。它的 case id 是 label 后接 dtype 取值，如 `llama-8b-prefill-float16`，nightly 与本站的 benchmark 行用的就是这个 id。

在代码里读 spec：

```python
from tileops.manifest import load_manifest, load_workloads

ops = load_manifest()                      # every entry, merged
list(ops["RMSNormFwdOp"]["signature"]["inputs"])  # ['x', 'weight']
load_workloads("RMSNormFwdOp")             # that op's workload rows
```

## 写一份新 spec {#writing-a-spec}

五步，每一步写完都能立即检查。

1. **起名，定 family。** 键是类名，条目写进 `spec/<family>.yaml`。
2. **写签名。** 会变的轴长、形状、dtype 都在 `forall` 里声明，每个张量的 `dtype` 与 `shape` 用这些类型变量来写。`params` 是算子 `__init__` 的参数列表，不含代码自己管的执行策略参数（`target`、`kernel_map`、`tune`）。可选输入排在必选输入之后。按参考 API 支持的来声明，不按当前 kernel 支持的来声明。
3. **写约束。** `shape_rules` 写关于类型变量取值的谓词，如 `H % G == 0`；派生的量写成 `let`；由开关选定的形状写成类型族。规则不读张量（`x.shape`、`x is None`），是否传入写成 `present(x)`。
4. **写 `workloads`。** 每一行恰好给出没有生成器能确定的类型变量、每个没有默认值的构造参数、`some`（这次传入的可选张量）、`dtype_cases`（条目有 `DType` 类型变量时才写；dtype 参数按参数写）与 `label`。`implemented` 条目的每个可选张量，至少一行传、至少一行不传。label 是 case id 的一部分，而 case id 是 nightly 历史数据的键，改 label 会让历史断开。
5. **写 `roofline`。** 用同一组类型变量写内联的 `flops`（访存不是「每个张量读或写一次」时再写 `bytes`），或者写一个 `func`，由它从检查过的调用算出两者。

接口先于实现落地时写 `status: spec-only`，这时需要读代码的检查都跳过；改成 `implemented` 后这些检查全部打开。

## 四种常见写法

### 由开关选定形状

`GemmFwdOp` 的两个布局开关决定 `a` 的哪个轴是 M。`types` 里的类型族写出两种情况，两个输入各自套用：

```yaml
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
```

### 可选输入作开关

`GroupNormFwdOp` 的 affine 由两个可选输入表达，旁边不再有 `affine` 开关：一件事只在一处声明。workload 行通过 `some` 传入它们，roofline 用 `present` 读是否传入：

```yaml
  signature:
    forall: {B: Dim, C: Dim, L: Shape, T: "DType[float32 | float16 | bfloat16]"}
    params:
      num_groups: {type: int}
      eps: {type: float, default: 1.0e-05}
    inputs:
      x: {dtype: T, shape: "[B, C, *L]"}
      weight: {dtype: T, shape: "[C]", optional: true}
      bias: {dtype: T, shape: "[C]", optional: true}
    outputs:
      output: {dtype: T, shape: "[B, C, *L]"}
    shape_rules:
      - "num_groups > 0 and C % num_groups == 0"
      - "B * (C // num_groups) * prod(L) != 1"
  workloads:
    - {B: 8, C: 128, L: [32, 32], num_groups: 32, dtype_cases: [{T: float16}], label: image}
    - {B: 8, C: 128, L: [32, 32], num_groups: 32, some: [weight, bias],
       dtype_cases: [{T: float16}], label: image-affine}
  roofline:
    flops: "(5 + (1 if present(weight) else 0) + (1 if present(bias) else 0)) * B * C * prod(L)"
```

（节选：条目里还有更多行和 dtype。）算子可以按可选输入是否传入来分派，不能读张量内容来决定。

### 被写入的输入

`SSDDecodeFwdOp` 的 `state` 每个 decode 步都被原地写回，所以它标 `mutated: true`，仍然是输入；调用只返回 `y_out`。生成的 operator 声明写入的参数，恰好是标了 `mutated` 的那些输入。

```yaml
    inputs:
      A: {dtype: float32, shape: "[H, P, N]"}
      dt: {dtype: float32, shape: "[B, H, P]"}
      x: {dtype: T, shape: "[B, H, P]"}
      B_in: {dtype: T, shape: "[B, G, N]"}
      C_in: {dtype: T, shape: "[B, G, N]"}
      state: {dtype: float32, shape: "[B, H, P, N]", mutated: true, contiguous: true}
    outputs:
      y_out: {dtype: float32, shape: "[B, H, P]"}
```

调用方可以自备的输出改标 `buffer: out`：`forward` 因此多一个 `out` 参数，算子写入它并返回它。

### 元数据张量的内容约束

偏移或长度这类张量，取值来自 `values` 里的生成器，内容须满足的条件写在 `requires` 里。下例取自 `GroupedQueryAttentionVarlenFwdOp`，其中 `q_lens` 是 `Seq[Int]` 类型变量，`T_q` 是 query 的总长度：

```yaml
cu_seqlens_q: {dtype: int32, shape: "[B + 1]", values: "prefix_sum(q_lens)",
               requires: ["prefix_offsets(T_q)"]}
```

`B` 由生成出的张量解出，所以一行给的是 `q_lens`，不是 `B`。

## 规则速查

**签名**

- **顺序即位置。** `params` 的顺序是 `__init__` 的参数顺序，`inputs` 是 `forward` 的参数顺序，`outputs` 是返回值顺序；调换顺序是不兼容的改动。
- **照参考实现写。** dtype 与参数依照权威的参考实现，不照当前代码；代码与之不符时改代码，改好之前条目标 `spec-only`。
- **同名即相等。** 形状相同的张量写同一个形状；不是「名字相等」的关系写成约束或 `let`。

**约束与是否传入**

- **约束读类型变量，不读张量。** 不写 `x.shape`、`x is None`、`isinstance`；是否传入写成 `present(x)`。
- **是否传入可以当开关，内容不可以。** 算子可以按参数与张量是否传入选择实现，张量内容只作计算输入。
- **可选输入两边都要测。** 传与不传各需一行，按输入逐个计，不按组合计。

**输出**

- **输出数量固定。** 一个条目在每次调用上输出都相同；返回值随开关变化的算子拆成两个条目。
- **被写入的输入仍是输入。** 它标 `mutated: true`，不列进 `outputs`。

## spec 校验器 {#spec-validator}

校验由 [`scripts/validate_manifest.py`](https://github.com/tile-ai/TileOPs/blob/main/scripts/validate_manifest.py) 执行，写完一份 spec 就可以立刻跑：

```bash
python scripts/validate_manifest.py                           # every entry
python scripts/validate_manifest.py --check-op RMSNormFwdOp   # one entry
python scripts/validate_manifest.py --levels schema,signature # skip the benchmark scan
```

| 级别 | 检查什么 |
| --- | --- |
| `schema` | 顶层字段、`family`、`ref_api`、`composition`、`roofline`，以及 `types.yaml` |
| `signature` | 签名在其判别量每种组合上的检查、每个 workload 行能否实例化、副作用声明；`implemented` 条目还要对照类的 `__init__`、`forward` 以及声明的 kernel 与子算子 |
| `bench` | 每个 benchmark 从 manifest 取调用、从算子取 roofline |

`spec-only` 条目跳过需要读代码的检查。kernel 选择、多 kernel 的执行顺序、累加 dtype、workspace、tile 尺寸与 autotune 配置不在 manifest 里，校验器也就看不到。
