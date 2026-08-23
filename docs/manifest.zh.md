# 读写 manifest

传统算子库以实现为中心：算子先被人逐个写出来、逐个调优，它支持什么形状、什么 dtype、跑多快，都由实现事后说明，文档是追述。TileOPs 的组织方式相反 —— 算子的规格先声明，实现由规格推导。承载这份规格的是 [`src/tileops/manifest/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/manifest) 下的条目。

**一个算子有了条目，它就成为整个系统的数据输入。** 各个环节读的是同一份声明，而不是各自去读实现：

| 谁消费 | 读条目里的什么 | 产出 |
| --- | --- | --- |
| 代码生成 | `signature`、`shape_rules` | 算子层的参数校验、形状推导、kernel 的调用签名 |
| 正确性测试 | `ref_api`、`workloads` 的 dtype | 与参考实现在每个 workload 上逐个比对数值 |
| 性能测量 | `workloads` | 每晚在这些形状上测得的 device time |
| roofline | `roofline` 的变量与公式 | 一次调用的计算量与访存量，效率的分母 |
| 编译契约 | `torch_compile_fullgraph` | CI 里一条要求 `fullgraph=True` 能编过的测试 |
| 文档 | 全部字段 | 支持矩阵、算子清单、API 参考 |
| [校验器](https://github.com/tile-ai/TileOPs/blob/main/scripts/validate_manifest.py) | 全部字段 | 分五级检查声明与实现是否一致，见[怎么写一条条目](#怎么写一条条目) |

**没有条目的算子，上面每一行都不会发生**：没有生成的校验、没有比对、没有性能数据，CI 也不会为它拦下任何回退。**为一个算子写 manifest 不是补文档，而是把它接进这条数据流。**{ .keystone }

本页分五部分：（1）一条条目由什么构成；（2）怎么读一条已有的条目；（3）怎么写一条新的；（4）构造时承诺的维度；（5）需要记住的规则。最后用六个真实条目覆盖不同的写法。

## 声明在前，实现在后

**方向只有一个：manifest 写在前，实现跟在后。** 条目对照的是权威参考（多数是 PyTorch 的公开 API），不是从 TileOPs 现有代码反推出来的；算子、测试与 benchmark 由它生成，再由校验器对照它检查。这条方向决定了两件事：实现随时可以从 spec 重新生成，而反过来做不到；代码与文档不一致时，以 manifest 为准。

**写了条目，就进入 CI 的保护范围。** 条目里的每个字段都有确定的消费者：

| 字段 | 谁读它 | 不一致时 |
| --- | --- | --- |
| `signature.inputs` / `outputs` | 算子层的校验、`_infer_output_shapes`、后端的 `build_kernel` 签名 | 校验器的 L1 报错：参数名或顺序与 `forward` 不符 |
| `signature.shape_rules` | 算子层的形状校验 | L2 报错：表达式不是合法的 Python，或与 `_infer_output_shapes` 的结论不一致 |
| `signature.dtype` 与 `dtype_combos` | 算子层的 dtype 校验 | L3 报错：dtype 名不存在，或与 `_validate_dtypes` 不一致 |
| `workloads` | benchmark 的形状与 dtype | L4 报错：benchmark 没有经 `load_workloads` 取形状 |
| `roofline` | `op.eval_roofline()`、性能报告里的效率 | 变量绑定不上就报错 |
| `torch_compile_fullgraph` | CI 的编译契约检查 | 声明了却没有登记编译测试，`compile-contract-gate` 这项检查失败 |
| `source` | 校验器定位 kernel、算子、测试与 benchmark 文件 | 路径不存在即报错 |

**没有条目的算子，这些检查一条都不会跑。** 所以给一个算子写 manifest 不是补文档，而是把它放进自动化的检查范围。

## 一条条目的构成

一个 family 一个 YAML 文件（大的 family 可以分片），文件顶层是 `算子名 → 条目` 的映射。加载时所有文件合并，同名条目重复即报错。

条目的键就是算子的 Python 类名：`{名字}{方向}Op`，方向取 `Fwd` 或 `Bwd`，同一个算子两个方向都在 manifest 里时方向必填。校验器要求 `cls.__name__` 与键逐字符相同，不做任何大小写或缩写的推断。

| 字段 | 必填 | 内容 |
| --- | --- | --- |
| `family` | 是 | 所属 family，决定条目写在哪个文件里 |
| `ref_api` | 是 | 权威参考的完整名字，例如 `torch.nn.functional.rms_norm`；没有对应实现时写 `"none"` |
| `status` | 是 | `spec-only` 或 `implemented`，决定校验跑到哪一级 |
| `torch_compile_fullgraph` | 否 | 只能写字面 `true`；不承诺就省略，写 `false` 是非法的 |
| `signature` | 是 | 算子接口本身，见下 |
| `workloads` | 是 | benchmark 用的形状与 dtype |
| `roofline` | 是 | 性能模型 |
| `source` | 是 | kernel、算子、测试、benchmark 的路径与 kernel 名映射 |

`signature` 之下有六个子字段：

| 子字段 | 内容 |
| --- | --- |
| `inputs` | 张量输入，`名字 → {dtype, shape?, constraints?, optional?, mutated?, layout?}` |
| `outputs` | 张量输出，字段同上，但输出的形状必须写全 |
| `params` | 非张量参数，`名字 → {type, default?, kw_only?}` |
| `shape_rules` | 形状关系，写成 Python 表达式的字符串 |
| `dtype_combos` | 跨张量的 dtype 组合，仅在支持的集合是各自取值域乘积的真子集时才写 |
| `static_dims` | 用户在构造算子时就承诺的维度值，只用于任意 rank 的算子 |

**`inputs`、`outputs`、`params` 的键顺序就是签名位置**，所以调整顺序是破坏性改动，读取时必须用保序的解析器。

## 怎么读一条条目

按四步读，每一步回答一个问题。

1. **`inputs` 与 `outputs`** —— 调用要传哪些张量、各自的 dtype 取值域，返回什么。
2. **`params`** —— 有哪些非张量参数。它们放在构造还是调用，由参考 API 决定，manifest 不编码这个区分；带 `default` 的参数在算子里也必须有同名默认值。
3. **`shape_rules`** —— `shape` 表达不了的约束都在这里，包括维度整除、`dim` 的取值范围、输出形状的推导。
4. **`workloads`** —— 这个算子被实测过哪些形状与 dtype。性能数据页上的每一行都出自这里。

读 dtype 的四种写法：

| 写法 | 含义 |
| --- | --- |
| `float16 \| bfloat16` | 取值域是这几种之一 |
| `same_as(x)` | 运行时与 `x` 的 dtype 相同。它只说 dtype，不说形状，也不为组合数增加一个维度 |
| `promote_int_to_float(x)` | `x` 是整数类型时结果为 `float32`，否则等同 `same_as(x)`。只允许出现在 `outputs` |
| `dtype_combos` | 逐条列出支持的跨张量组合。不写表示各自取值域的任意组合都支持 |

读形状先看有没有 `shape`：

- **有 `shape`** —— rank 固定，维度名写出来，例如 `"[B, M, K]"`。**同名维度表示相等**：两个张量都写 `K`，它们那一维必须一样长。
- **没有 `shape`** —— rank 任意，约束全在 `params` 与 `shape_rules` 里。

程序化读取用 `tileops.manifest`：

```python
from tileops.manifest import load_manifest, load_workloads

ops = load_manifest()                      # 合并后的全部条目
entry = ops["RMSNormFwdOp"]
entry["signature"]["inputs"].keys()        # dict_keys(['x', 'weight'])

load_workloads("RMSNormFwdOp")             # 该算子的 workload 列表
```

## 怎么写一条条目

按五步写，每一步都能立刻校验。

1. **起名，选 family。** 键是算子的类名，条目写进 `family` 对应的那个文件。
2. **写 `signature`。** 张量进 `inputs` / `outputs`，非张量进 `params`；顺序按调用顺序排，可选输入排在必填输入之后。dtype 要写全参考 API 支持的范围，而不是当前 kernel 支持的范围。
3. **写 `shape_rules`。** 输出形状必须由 `shape` 与 `shape_rules` 完全确定。涉及 `dim` 的算子用 [`shape_rules.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/manifest/shape_rules.py) 里的 `dim_range_validity`、`reduced_shape` 等辅助函数 —— 算子层调用的是同一批函数，所以两边不会各说一套。
4. **写 `workloads`。** 单张量输入的算子，形状键必须是 `{输入名}_shape`，其余键只能是 `params` 的名字或保留的 `dtypes` / `label`。
5. **写 `roofline` 与 `source`。**

接口先落地、实现在后时，条目写 `status: spec-only`，校验只跑 L0（结构检查）；改成 `implemented` 之后 L0–L4 全跑。

校验由 [`scripts/validate_manifest.py`](https://github.com/tile-ai/TileOPs/blob/main/scripts/validate_manifest.py) 执行：

```bash
python scripts/validate_manifest.py                       # 全部条目
python scripts/validate_manifest.py --check-op SoftmaxFwdOp   # 单条，强制跑满五级
```

五级检查各管一件事：L0 结构，L1 签名与 `forward` 的参数名与顺序，L2 `shape_rules` 的语法与形状推导的一致性，L3 dtype 名与 dtype 校验的一致性，L4 benchmark 是否经 `load_workloads` 取形状。

## 构造时承诺的维度：`static_dims`

`static_dims` 声明用户在构造算子实例时就承诺下来的维度值。它只用于任意 rank 的算子 —— 固定 rank 的算子从 `shape` 就能拿到维度。

```yaml
static_dims:
  N: "x.shape[dim]"
```

**表达式是调用时的校验规则，不是构造时的推导。** 两个时点共用一份契约：

| 时点 | 发生什么 |
| --- | --- |
| `__init__` | 承诺点。用户给出的值存到 `self` 上，表达式不求值 —— 此时还没有张量 |
| `forward` | 校验点。表达式对实际张量求值，结果必须等于承诺的值，不等就报错 |

四条规则：

- 每个键都是 `__init__` 的**必填**关键字参数，不能有默认值。承诺的值必须由用户在构造时给出。
- 表达式必须是**单轴引用**，形如 `<张量>.shape[<常量或参数名>]`。多轴形式一概禁止，包括 `product(...)`、推导式与形状上的算术。
- 引用的张量名必须出现在 `signature.inputs` 里；轴名不是整数字面量时，必须是 `signature.params` 的名字。
- 键顺序决定这些关键字参数在生成的 `__init__` 中的顺序。

表达式可以引用任意一个输入张量，不限于第一个。`torch.nn.functional.linear` 的 `out_features` 就只能绑到 `weight` 上 —— 用 `input.shape` 写不出等价的表达式：

```yaml
LinearFwdOp:
  signature:
    inputs:
      input: {dtype: "float16 | bfloat16"}
      weight: {dtype: "same_as(input)"}
      bias: {dtype: "same_as(input)"}
    outputs:
      output: {dtype: "same_as(input)"}
    static_dims:
      in_features: "input.shape[-1]"
      out_features: "weight.shape[0]"
    shape_rules:
      - "weight.shape == (out_features, in_features)"
      - "bias.shape == (out_features,)"
      - "output.shape == input.shape[:-1] + (out_features,)"
```

**`static_dims` 为空是合法的**，典型情形是接受 `dim=None` 的归约 —— 归约的范围取决于整个输入形状，不是用户给出的超参数。但此时算子作者**必须覆写 `_cache_key`**：默认实现按完整的输入形状做键，正确但在动态形状下每个不同形状都会重新编译。基类在这种情形下会发一次运行时警告。

## 需要记住的规则

| 规则 | 内容 |
| --- | --- |
| 顺序即位置 | `inputs`、`outputs`、`params` 的键顺序就是签名位置，调整顺序是破坏性改动 |
| 接口写全 | `params` 覆盖参考 API 的全部参数，即使当前 kernel 只支持默认值 |
| 输出形状必须完全确定 | 每个输出的形状由 `shape` 与 `shape_rules` 完全确定；输入可以不写 `shape` |
| 不做形状别名 | 每个张量各自声明形状，关系用同名维度或 `shape_rules` 表达 |
| `optional` 只给输入 | 张量输入用 `optional: true` 表达可选，参数用 `default` |
| 输出数量固定 | 一条条目的输出名字与数量对每次调用固定。返回值随开关变化的算子拆成两条条目 |
| 一条条目一种内存序 | 内存序不同就是两条条目 —— 换了内存序，同一个轴的含义就变了 |
| 参数不选形状 | 参数可以决定某一维的大小，不能决定张量取哪一种形状 |
| 存在性是开关，取值不是 | 算子可以读「某个可选输入在不在」来选实现，不能读张量的内容来选实现 |
| 可选输入两侧都要测 | 每个可选输入的传与不传，各至少要有一行 workload 命中。逐个输入数，不按组合数 |

## 六个例子

六个都是仓里的真实条目，各覆盖一种写法：（1）固定 rank；（2）任意 rank；（3）参数决定输出形状；（4）可选输入构成一个开关；（5）会被写入的输入；（6）输出 dtype 随输入提升。

### 1. 固定 rank，同名维度表示相等

`BmmFwdOp` 的三个张量都写出了 `shape`，`B` 与 `K` 在两个输入里同名，因此必须相等；`shape_rules` 再补一条 kernel 要求的整除条件。

```yaml
BmmFwdOp:
  ref_api: torch.bmm
  family: gemm
  status: implemented
  signature:
    inputs:
      a: {dtype: "float16 | bfloat16", shape: "[B, M, K]"}
      b: {dtype: "same_as(a)", shape: "[B, K, N]"}
    outputs:
      d: {dtype: "same_as(a)", shape: "[B, M, N]"}
    shape_rules:
      - "a.shape[0] == b.shape[0]"
      - "a.shape[2] == b.shape[1]"
      - "a.shape[2] % 16 == 0"
      - "d.shape == (a.shape[0], a.shape[1], b.shape[2])"
```

### 2. 任意 rank，约束写进 shape_rules

`RMSNormFwdOp` 不限制输入的 rank，归一化的轴由参数 `normalized_shape` 给出，因此三个张量都不写 `shape`，关系全部落在 `shape_rules` 里。

```yaml
  signature:
    inputs:
      x: {dtype: "float16 | bfloat16"}
      weight: {dtype: "same_as(x)"}
    outputs:
      output: {dtype: "same_as(x)"}
    params:
      normalized_shape: {type: "list[int] | tuple[int, ...]"}
      eps: {type: "float | None", default: null}
    shape_rules:
      - "len(normalized_shape) > 0"
      - "tuple(x.shape[-len(normalized_shape):]) == tuple(normalized_shape)"
      - "weight.shape == tuple(normalized_shape)"
      - "output.shape == x.shape"
```

### 3. 参数决定输出形状

`GemmFwdOp` 的两个布局标志决定哪一维是 M、哪一维是 N，所以输出形状是一条含条件表达式的规则，而不是一个 `shape` 声明。

```yaml
  signature:
    inputs:
      a: {dtype: "float16 | bfloat16"}
      b: {dtype: "same_as(a)"}
    outputs:
      d: {dtype: "same_as(a)"}
    params:
      trans_a: {type: bool, default: false}
      trans_b: {type: bool, default: true}
    shape_rules:
      - "d.shape == ((a.shape[1] if trans_a else a.shape[0]), (b.shape[0] if trans_b else b.shape[1]))"
```

### 4. 可选张量输入构成一个开关

`GroupNormFwdOp` 的 affine 由 `weight` 与 `bias` 两个可选输入表达。它们是同一个开关的两半，这层关系写在 `shape_rules` 里，与形状约束同等对待；引用可选输入的规则自己写出 guard，不依赖「缺席时这条规则不适用」的默契。

```yaml
  signature:
    inputs:
      x: {dtype: "float32 | float16 | bfloat16"}
      weight: {dtype: "same_as(x)", optional: true}
      bias: {dtype: "same_as(x)", optional: true}
    outputs:
      output: {dtype: "same_as(x)"}
    params:
      num_groups: {type: int}
      eps: {type: float, default: 1.0e-05}
    shape_rules:
      - "(weight is None) == (bias is None)"
      - "weight is None or weight.shape == (x.shape[1],)"
      - "bias is None or bias.shape == (x.shape[1],)"
      - "x.shape[1] % num_groups == 0"
      - "output.shape == x.shape"
```

不另设 `affine: bool` 参数：张量在不在本身就是这个开关，再加一个参数就有两处记同一件事，且能互相矛盾。

workloads 一侧不传、一侧传，两侧都要有行命中 —— `weight_shape` 与 `bias_shape` 两个键在场就是传，缺席就是不传：

```yaml
  workloads:
    - {x_shape: [8, 128, 32, 32], num_groups: 32, dtypes: [float16, bfloat16], label: "image-g32"}
    - {x_shape: [8, 128, 32, 32], num_groups: 32, weight_shape: [128], bias_shape: [128],
       dtypes: [float16, bfloat16], label: "image-g32-affine"}
```

### 5. 会被写入的输入

算子可能写入的张量输入声明 `mutated: true`。`SSDDecodeFwdOp` 的 `state` 就是这样：解码一步之后新的状态直接写回这个张量，函数只返回 `y_out`。

```yaml
SSDDecodeFwdOp:
  ref_api: "none"
  family: mamba
  status: implemented
  signature:
    inputs:
      A: {dtype: "float32", shape: "[H, P, N]"}
      dt: {dtype: "float32", shape: "[B, H, P]"}
      x: {dtype: "float16 | bfloat16", shape: "[B, H, P]"}
      B_in: {dtype: "same_as(x)", shape: "[B, G, N]"}
      C_in: {dtype: "same_as(x)", shape: "[B, G, N]"}
      state: {dtype: "float32", shape: "[B, H, P, N]", mutated: true}
    outputs:
      y_out: {dtype: "float32", shape: "[B, H, P]"}
    params: {}
    shape_rules:
      - "H % G == 0"
```

三点需要注意：

- **被写入的输入仍然是输入。** 输出的数量不变，返回值也不与它别名 —— `state` 不出现在 `outputs` 里。
- 声明的集合与算子注册的算子边界一致：`mutates_args` 指名的输入，恰好是标了 `mutated: true` 的那些。
- 如果连续性归一时复制过这个张量，算子在 kernel 启动之后要把结果写回原张量。

### 6. 输出 dtype 随输入提升

`torch.reciprocal` 接受整数输入并返回 `float32`，浮点输入原样返回。这种提升写成 `promote_int_to_float`，算子层照它做转换，校验器展开成具体集合后再对照 `_validate_dtypes`。

```yaml
ReciprocalFwdOp:
  ref_api: "torch.reciprocal"
  signature:
    inputs:
      input: {dtype: "float16 | bfloat16 | float32 | int8 | int16 | int32 | int64 | uint8"}
    outputs:
      output: {dtype: "promote_int_to_float(input)"}
```

## 校验器不做的事

manifest 的校验是语法与一致性检查，不是求值：

- `shape_rules` 只做语法检查，不求值、不枚举传法，也不区分「管传没传」与「管形状」两类规则。
- 传错的调用由算子自己的运行时检查拦截。传了 `weight` 却不传 `bias`，manifest 校验通过，报错来自 [`GroupNormFwdOp.forward`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/ops/norm/group_norm.py)。
- manifest 不描述 kernel 的执行细节：多 kernel 的执行顺序、累加 dtype、持久状态、tile 尺寸、autotune 配置都不在其中。

完整的字段规范与全部规则见 [Op Manifest](design/manifest.md)。
