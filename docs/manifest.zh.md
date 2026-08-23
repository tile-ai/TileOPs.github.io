# 读写 manifest

传统算子库以实现为中心：算子逐个写、逐个调优，支持哪些形状、哪些 dtype、跑多快，都由实现事后说明，文档写的是追述。

TileOPs 的组织方式相反：算子的规格先声明，实现由规格推导。每个算子的规格称为它的 **spec**，写在 [`src/tileops/manifest/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/manifest) 下的 YAML 文件里；这些文件合起来就是 manifest。

**一个算子有了 spec，就成为整个系统的数据输入。** 各个环节读同一份声明，而不是各自去读实现：

| 谁消费 | 读 spec 里的什么 | 产出 |
| --- | --- | --- |
| codegen，即生成算子与 kernel 的 agent | `signature`、`shape_rules` | 算子层的参数校验、形状推导、kernel 的调用签名 |
| [pytest](https://github.com/tile-ai/TileOPs/tree/main/tests) | `ref_api`、`workloads` 的 dtype | 与参考实现在每个 workload 上逐个比对数值 |
| [每晚的 benchmark](https://github.com/tile-ai/TileOPs/tree/main/benchmarks) | `workloads` | 这些形状上测得的 device time |
| [roofline](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/perf) | `roofline` 的变量与公式 | 一次调用的计算量与访存量，效率的分母 |
| CI 的 `compile-contract-gate` | `torch_compile_fullgraph` | 要求 `fullgraph=True` 能编过的那条测试 |
| 本文档站 | 全部字段 | 支持矩阵、算子清单、API 参考 |
| CI 的 [spec 校验器](https://github.com/tile-ai/TileOPs/blob/main/scripts/validate_manifest.py) | 全部字段 | 分五级检查声明与实现是否一致，见[写一份新 spec](#writing-a-spec) |

**每一行都以 spec 为前提**：没有 spec，就没有生成的校验、没有数值比对、没有性能数据，CI 也拦不下任何回退。**为一个算子写 manifest 不是补文档，而是把它接进这条数据流。**{ .keystone }

本页依次讲：一份 spec 的构成、怎么读一份已有的 spec、怎么写一份新的、`static_dims`（构造时承诺的维度）、可选输入（写 spec 时分歧最多的地方）。之后是六个实例分析、提交前对照用的规则速查，以及 spec 校验器查什么、不查什么。

## spec 驱动的生成与检查

写了 spec，这个算子就进入 CI 的保护范围。现有的算子、测试与 benchmark 都由 spec 生成，三处各读其中一部分：

- **算子层**的参数校验与形状推导照 `signature` 写。
- **测试**的比对对象与 dtype 取自 `ref_api` 与 `workloads`。
- **benchmark** 的形状由 `load_workloads` 读出来。

生成之后，spec 校验器逐级对照 spec 检查这些代码，声明与实现对不上就报错；哪个字段以什么形式报出来，见[spec 校验器](#spec-validator)。

**没有 spec 的算子不受这些检查约束。** 所以新增算子先写 spec：spec 落地之后，后续每次改动都由 CI 对照它检查。

## 一份 spec 的构成

一个 family 一个 YAML 文件（大的 family 可以分片），顶层是 `算子名 → spec` 的映射。加载时所有文件合并，同名算子重复即报错。

spec 的键就是算子的 Python 类名：`{名字}{方向}Op`，方向取 `Fwd` 或 `Bwd`；两个方向都在 manifest 里时方向必填。校验器要求 `cls.__name__` 与键逐字符相同，不做大小写或缩写的推断。

| 字段 | 必填 | 内容 |
| --- | --- | --- |
| `family` | 是 | 所属 family，决定这份 spec 写在哪个文件里 |
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

## 读一份 spec

四步，每步回答一个问题。

1. **`inputs` 与 `outputs`** —— 调用要传哪些张量、各自的 dtype 取值域，返回什么。
2. **`params`** —— 有哪些非张量参数。放在构造还是调用由参考 API 决定，manifest 不编码这个区分；带 `default` 的参数在算子里也必须有同名默认值。
3. **`shape_rules`** —— `shape` 表达不了的约束都在这里：维度整除、`dim` 的取值范围、输出形状的推导。
4. **`workloads`** —— 实测过哪些形状与 dtype。性能数据页上的每一行都出自这里。

读 dtype 的四种写法：

- `float16 | bfloat16` —— 取值域是这几种之一。
- `same_as(x)` —— 运行时与 `x` 的 dtype 相同。只说 dtype，不说形状，也不为组合数增加一个维度。
- `promote_int_to_float(x)` —— `x` 是整数类型时结果为 `float32`，否则同 `same_as(x)`。只允许出现在 `outputs`。
- `dtype_combos` —— 逐条列出支持的跨张量组合。不写表示各自取值域的任意组合都支持。

读形状先看有没有 `shape`：

- **有 `shape`** —— rank 固定，维度名写出来，如 `"[B, M, K]"`。**同名维度表示相等**：两个张量都写 `K`，那一维就必须一样长。
- **没有 `shape`** —— rank 任意，约束全在 `params` 与 `shape_rules` 里。

程序化读取用 `tileops.manifest`：

```python
from tileops.manifest import load_manifest, load_workloads

ops = load_manifest()                      # 合并后的全部 spec
spec = ops["RMSNormFwdOp"]
spec["signature"]["inputs"].keys()         # dict_keys(['x', 'weight'])

load_workloads("RMSNormFwdOp")             # 该算子的 workload 列表
```

## 写一份新 spec {#writing-a-spec}

五步，每步都能立刻校验。

1. **起名，选 family。** 键是算子的类名，spec 写进 `family` 对应的那个文件。
2. **写 `signature`。** 张量进 `inputs` / `outputs`，非张量进 `params`；按调用顺序排，可选输入排在必填输入之后。dtype 写参考 API 支持的全部范围，不是当前 kernel 支持的范围。
3. **写 `shape_rules`。** 输出形状必须由 `shape` 与 `shape_rules` 完全确定。涉及 `dim` 的算子用 [`shape_rules.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/manifest/shape_rules.py) 里的 `dim_range_validity`、`reduced_shape` 等辅助函数 —— 算子层调的是同一批函数，两边不会各说一套。
4. **写 `workloads`。** 单张量输入的算子，形状键必须是 `{输入名}_shape`，其余键只能是 `params` 的名字或保留的 `dtypes` / `label`。
5. **写 `roofline` 与 `source`。**

接口先落地、实现在后，spec 写 `status: spec-only`，只跑 L0；改成 `implemented` 之后五级全跑。每级查什么、校验器管不到什么，见最后一节 [spec 校验器](#spec-validator)。

## `static_dims`

`static_dims` 声明用户在构造算子实例时就承诺的维度值，只用于任意 rank 的算子 —— 固定 rank 的算子从 `shape` 就能拿到维度。

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

- 每个键都是 `__init__` 的**必填**关键字参数，不能有默认值 —— 承诺的值由用户在构造时给出。
- 表达式必须是**单轴引用**，形如 `<张量>.shape[<常量或参数名>]`；多轴形式一概禁止，包括 `product(...)`、推导式与形状上的算术。
- 引用的张量名必须出现在 `signature.inputs` 里；轴名不是整数字面量时，必须是 `signature.params` 的名字。
- 键顺序决定这些关键字参数在生成的 `__init__` 中的顺序。

单轴引用这一条最容易写错，两种被拒的写法都出自同一个原因：在 `forward` 里没法逐轴校验。

```yaml
static_dims:
  in_features: "input.shape[-1]"        # 接受
  out_features: "weight.shape[0]"       # 接受：可以引用任意一个输入张量
# numel: "product(x.shape)"             # 不接受：多轴
# last: "x.shape[x.ndim - 1] * 2"       # 不接受：形状上的算术
```

引用哪个张量不受限制。`torch.nn.functional.linear` 的 `out_features` 只能绑到 `weight` 上，用 `input.shape` 写不出等价的表达式：

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

**`static_dims` 为空是合法的。** 典型情形是接受 `dim=None` 的归约：归约范围取决于整个输入形状，不是用户给出的超参数，没有什么可以在构造时承诺。

为空的时候，算子作者**必须覆写 `_cache_key`**。默认实现按完整输入形状做键，结果正确，但动态形状下每来一个新形状都要重新编译；基类遇到这种情形会发一次运行时警告。

```python
class SumFwdOp(Op):
    def _cache_key(self, x_shape):
        return (math.prod(x_shape),)   # 全归约：元素数相同即共享 kernel
```

## 可选输入 {#optional-inputs}

一个输入可以不传，`shape_rules`、`workloads`、`roofline` 与 dtype 声明都要跟着改，所以这里是写 spec 时分歧最多的地方。下面十条各回答一个抉择，示例都取自仓里的真实 spec：

- **第 1、2 条**：可选输入是什么，调用时怎样算传了。
- **第 3、4 条**：写 spec 时的两个抉择 —— 能不能按存在性派发，该不该另加一个 `bool`。
- **第 5、6 条**：`shape_rules` 怎么跟着写。
- **第 7 到 9 条**：`workloads`、`roofline`、dtype 声明各自怎么跟着改。
- **第 10 条**：输出的数量不随开关变化。

**1. 可选输入和可选参数怎么区分？**

两者用不同的字段表达，因为可选的含义本来就不同：

- **张量输入**用 `optional: true`。张量要么在要么不在，没有中间状态。
- **参数**用 `default`。参数总有一个值，调用方没给就取默认值。

另有两条边界：

- `optional` 只允许出现在 `inputs` 下。**输出不能是可选的** —— 调用方需要事先知道自己会收到几个返回值。
- 调用方自己准备好、交给算子写进去的输出缓冲（`out=`）属于参数，不是可选输入。

```yaml
# MoePermuteNopadFwdOp
inputs:
  hidden_states: {dtype: "float16 | bfloat16"}
  topk_ids: {dtype: "int32"}
  expert_map: {dtype: "int32", optional: true}   # 张量：optional
params:
  num_experts: {type: int}
  num_experts_local: {type: int}                 # 参数若可选，用 default
```

**2. 调用时怎样算「传了这个可选输入」，它排在参数表的哪个位置？**

以 `Mamba2FwdOp` 为例。它的 spec 声明了五个必填输入和两个可选输入：

```yaml
inputs:
  x: {dtype: "float16 | bfloat16", shape: "[B, S, H, P]"}
  dt: {dtype: "float32", shape: "[B, S, H]"}
  A: {dtype: "float32", shape: "[H]"}
  B: {dtype: "same_as(x)", shape: "[B, S, G, N]"}
  C: {dtype: "same_as(x)", shape: "[B, S, G, N]"}
  dt_bias: {dtype: "float32", shape: "[H]", optional: true}
  initial_states: {dtype: "float32", shape: "[B, H, P, N]", optional: true}
```

`forward` 的形参顺序照抄这份声明，两个可选输入的默认值是 `None`。调用处于是有四种写法：

```python
op = Mamba2FwdOp()

y, final = op(x, dt, A, B, C)                        # dt_bias、initial_states 都没传
y, final = op(x, dt, A, B, C, dt_bias=None)          # 与上一行完全相同
y, final = op(x, dt, A, B, C, dt_bias=bias)          # 传了 dt_bias
y, final = op(x, dt, A, B, C, initial_states=state)  # 传了 initial_states
```

四种写法只对应两种状态。三点可以从上面这段代码直接读出来：

- **判据是绑定之后的值** —— 为 `None` 就是没传。
- **第一行与第二行落到同一处。** 省略关键字与显式写 `None` 不是两个状态，因为缺席不是取值域里的一个值，`None` 只是它在 Python 层的拼写。
- **算子读到的也正是这个值。** `dt_bias is None` 为真，这次调用就没有 `dt_bias`。

参数表的位置由这个默认值决定：**可选输入必须排在必填输入之后。** `forward` 按 `inputs` 的声明顺序接收参数，而带默认值的参数不能排在必填参数之前。参考 API 把可选参数放在中间时，spec 按这条规则重排，不照抄参考签名里的位置。

**3. 算子能不能根据可选输入是否传入来选实现？**

能，这正是可选输入的用途之一。`MoePermuteNopadFwdOp` 的 `expert_map` 就是这样一个开关，它的 spec 声明如下：

```yaml
# Expert parallelism: supplied when this rank owns num_experts_local of the
# num_experts global experts, absent when it owns them all. Presence picks
# the scan kernel; the ids inside are read only at launch.
expert_map: {dtype: "int32", optional: true}
```

算子的 `forward` 里，派发只看这个参数在不在：

```python
# tileops/ops/moe/routed_expert/permute_nopad.py
self.used_expert_map = expert_map is not None

# A map-free kernel and a map-reading one are two different scans, so the
# argument's presence — not the local expert count — picks between them.
kernel = self._get_kernel(
    (hidden_states, topk_ids)
    if expert_map is None
    else (hidden_states, topk_ids, expert_map),
    ...
)
if expert_map is None:
    return kernel(hidden_states, topk_ids)
return kernel(hidden_states, topk_ids, expert_map)
```

这段代码读的是 `expert_map is None`，从不读 `expert_map` 里的值，可选输入的用法也就到此为止：**张量在不在可以决定用哪个 kernel，张量的内容不可以。** 形状与存在性在 spec 里有声明，内容没有；`expert_map` 里的 id 要等 kernel 启动之后才被读到。

**4. 一个能关掉的特性，声明成可选的 `inputs` 条目，还是声明成一个 `bool` 开关参数？**

声明成可选的 `inputs` 条目，不要再给它配一个 `bool`。

以 GroupNorm 的 affine 为例。参考 API 把它做成一个 `affine=True` 开关加一对权重，照抄过来就是这样一份 spec：

```yaml
# 不推荐：affine 与两个可选张量记的是同一件事
inputs:
  x: {dtype: "float32 | float16 | bfloat16"}
  weight: {dtype: "same_as(x)", optional: true}
  bias: {dtype: "same_as(x)", optional: true}
params:
  num_groups: {type: int}
  eps: {type: float, default: 1.0e-05}
  affine: {type: bool, default: true}
```

「这次调用要不要做 affine」是一件事，这份 spec 却把它记在了两处：`affine` 的值，以及 `weight` 有没有传。只要有两处，它们就可能对不上，而 spec 校验对下面两种情形都不会报错：

- `affine: true`，但 `weight` 没传 —— 算子被要求做 affine，手上却没有权重。
- `affine: false`，但传了 `weight` —— 传进来的张量没有任何一处会读它。

推荐的写法只留一处来源，`params` 里不出现 `affine`：

```yaml
# GroupNormFwdOp 的 spec
inputs:
  x: {dtype: "float32 | float16 | bfloat16"}
  weight: {dtype: "same_as(x)", optional: true}
  bias: {dtype: "same_as(x)", optional: true}
params:
  num_groups: {type: int}
  eps: {type: float, default: 1.0e-05}
```

算子里读 `weight is not None`，取代原来读 `affine` 的地方：行为不变，两种不一致都不再能被写出来。

这条判据一般成立：**一个 `bool` 参数的值总能由某个可选输入在不在推出来，就不要声明它。**

**5. 两个可选输入必须同时传或同时不传，写在哪里？**

写进 `shape_rules`，与形状约束同等对待，不新增字段：

```yaml
shape_rules:
  - "(weight is None) == (bias is None)"          # 同一个开关的两半
  - "not (min is None and max is None)"           # 至少传一个
  - "running_mean is None or not use_input_stats" # 存在性约束到某个参数的取值上
```

**6. 引用可选输入的规则怎么写？**

自己写出 guard，形如 `X is None or <用到 X 的判断>`；guard 写在使用之前即可，不必在最左。

```yaml
# MoePermuteNopadFwdOp
shape_rules:
  - "expert_map is None or expert_map.shape == (num_experts,)"   # 接受
# - "expert_map is not None and expert_map.shape == (num_experts,)"  # 不接受
```

`and` 形态不接受：`shape_rules` 的每一条都是必须成立的合取项，它在合法的缺席调用上求值为假，等于把合法调用判成违规。

**7. workloads 要写几行？**

每个可选输入的传与不传各至少一行。**逐个输入数，不按组合数** —— n 个可选输入是 2n 个状态，不是 2ⁿ 个。状态由键编码：`<输入名>_shape` 在场就是传，缺席就是不传。

```yaml
# MoePermuteNopadFwdOp：expert_map 两侧各有行
workloads:
  - {hidden_states_shape: [1, 7168], topk_ids_shape: [1, 8], num_experts: 384,
     num_experts_local: 384, dtypes: [bfloat16], label: "kimi-k2-decode"}
  - {hidden_states_shape: [1, 7168], topk_ids_shape: [1, 8], num_experts: 256,
     num_experts_local: 128, expert_map_shape: [256], dtypes: [bfloat16],
     label: "deepseek-v3-ep2-decode"}
```

即使这一族的行用标量维度给形状，传入那一侧也照样写上这个键 —— `GemmFp8FwdOp` 传 `bias` 的那行就在 `n: 2112` 旁边写 `bias_shape: [2112]`。

**8. roofline 怎么写才能算上可选输入？**

内联写法里，存在性是 `vars` 层的一个布尔量，`flops` 与 `bytes` 只引用那个 `vars` 名 —— 张量名在算术层根本不解析。

```yaml
# GroupNormFwdOp
roofline:
  vars:
    N: "x.shape[0]"
    C: "x.shape[1]"
    spatial_size: "product(x.shape[2:])"
    affine: "weight is not None"                       # 存在性放在 vars 层
  flops: "(5 if affine else 3) * N * C * spatial_size" # 算术层只引用 vars 名
```

公式需要可选输入自己的形状时，改用 `roofline: {func: ...}`，函数从实际调用里读得到。

**9. `dtype_combos` 和 `same_as()` 能引用可选输入吗？**

不能。这两处对每次调用都要成立，而缺席那次调用里这个名字没有指向。

```yaml
# 不接受：weight 是可选输入，下面两处都在指望它一定存在
inputs:
  x: {dtype: "float32 | float16", shape: "[B, S, H]"}
  weight: {dtype: "same_as(x)", shape: "[H]", optional: true}
  bias: {dtype: "same_as(weight)", shape: "[H]", optional: true}
dtype_combos:
  - {x: float16, weight: float16}
```

两处各错在一件事上：

- `dtype_combos` 的键必须每次调用都存在，`weight` 没传的那次调用凑不出这个组合。
- `same_as(weight)` 里的 ref 是可选输入，`weight` 没传时 `bias` 的 dtype 无从解析。

改法是把两处都挂到必填输入上：

```yaml
inputs:
  x: {dtype: "float32 | float16", shape: "[B, S, H]"}
  weight: {dtype: "same_as(x)", shape: "[H]", optional: true}
  bias: {dtype: "same_as(x)", shape: "[H]", optional: true}
dtype_combos:
  - {x: float16}
```

**10. 返回值数量随开关变化的算子怎么办？**

拆成两份 spec，或者把输出定死。一份 spec 的输出名字与数量对每次调用固定，否则调用方无法解包一个形状未知的返回值。`Mamba2FwdOp` 选的是后者，并在注释里记下这处与上游的偏离：

```yaml
# ... one declared deviation: final_states is always returned (fixed output
# arity; the Op protocol has no conditional-output mechanism), so the upstream
# return_final_states switch does not exist here.
outputs:
  y: {dtype: "float32", shape: "[B, S, H, P]"}
  final_states: {dtype: "float32", shape: "[B, H, P, N]"}
```

## 实例分析

六份真实 spec，各覆盖一种写法：固定 rank、任意 rank、参数决定输出形状、可选输入作开关、会被写入的输入、输出 dtype 随输入提升。

### 1. 固定 rank 与同名维度

`BmmFwdOp` 的三个张量都写出 `shape`，`B` 与 `K` 在两个输入里同名，因此必须相等；`shape_rules` 再补一条 kernel 要求的整除条件。

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

### 2. 任意 rank

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

### 4. 可选输入作开关

`GroupNormFwdOp` 的 affine 由 `weight` 与 `bias` 两个可选输入表达。它们是同一个开关的两半，这层关系写进 `shape_rules`，与形状约束同等对待。

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

这份 spec 是[可选输入](#optional-inputs)那一节的三条规则落到实处的样子：`params` 里没有 `affine`（第 4 条），三条引用可选输入的规则各自写出 guard（第 6 条），workloads 传与不传两侧都有行命中（第 7 条）。

最后一条对应的就是下面两行 —— `weight_shape` 与 `bias_shape` 在场就是传，缺席就是不传：

```yaml
  workloads:
    - {x_shape: [8, 128, 32, 32], num_groups: 32, dtypes: [float16, bfloat16], label: "image-g32"}
    - {x_shape: [8, 128, 32, 32], num_groups: 32, weight_shape: [128], bias_shape: [128],
       dtypes: [float16, bfloat16], label: "image-g32-affine"}
```

### 5. 被写入的输入

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

这份 spec 有三点要注意：

- **被写入的输入仍然是输入。** 输出的数量不变，返回值也不与它别名 —— `state` 不出现在 `outputs` 里。
- 声明的集合与算子注册的算子边界一致：`mutates_args` 指名的输入，恰好是标了 `mutated: true` 的那些。
- 如果连续性归一时复制过这个张量，算子在 kernel 启动之后要把结果写回原张量。

### 6. dtype 提升

`torch.reciprocal` 接受整数输入并返回 `float32`，浮点输入原样返回。这种提升写成 `promote_int_to_float`，算子层照它做转换，spec 校验器展开成具体集合后再对照 `_validate_dtypes`。

```yaml
ReciprocalFwdOp:
  ref_api: "torch.reciprocal"
  signature:
    inputs:
      input: {dtype: "float16 | bfloat16 | float32 | int8 | int16 | int32 | int64 | uint8"}
    outputs:
      output: {dtype: "promote_int_to_float(input)"}
```

## 规则速查

这十条前面都讲过，这里按各自约束什么分成四组，方便把写好的 spec 拿来逐条对一遍，再去跑下一节的校验器。

**签名**

- **顺序即位置。** `inputs`、`outputs`、`params` 的键顺序就是 `forward` 里的参数位置，因此调整顺序会一并改掉所有调用方的写法，是破坏性改动。
- **接口写全。** `params` 覆盖参考 API 的全部参数，即使当前 kernel 只支持其中的默认值 —— spec 描述的是这个算子的接口，不是这一版实现的能力。

**形状**

- **输出形状必须完全确定。** 每个输出的形状由 `shape` 与 `shape_rules` 一起唯一确定。输入可以不写 `shape`，输出不行。
- **不做形状别名。** 每个张量各自声明自己的形状，张量之间的关系用同名维度或 `shape_rules` 表达。
- **参数不选形状。** 参数可以决定某一维有多大，不能决定这个张量取哪一种形状；后者会让一份 spec 描述出两个不同的接口。

**可选与开关**

- **`optional` 只给输入。** 张量输入用 `optional: true` 表达可选，参数用 `default` 表达，两者不换用。
- **存在性是开关，取值不是。** 算子可以读「某个可选输入在不在」来选实现，不能读张量里的内容来选实现。
- **可选输入两侧都要测。** 每个可选输入的传与不传各至少要有一行 workload 命中，逐个输入数，不按组合数。

**输出与内存序**

- **输出数量固定。** 一份 spec 的输出名字与数量对每次调用都相同。返回值随开关变化的算子拆成两份 spec，否则调用方无法解包一个数量未知的返回值。
- **一份 spec 一种内存序。** 内存序不同就是两份 spec —— 换了内存序，同一个轴的含义就变了。

## spec 校验器 {#spec-validator}

校验由 [`scripts/validate_manifest.py`](https://github.com/tile-ai/TileOPs/blob/main/scripts/validate_manifest.py) 执行，写完一份 spec 就可以立刻跑：

```bash
python scripts/validate_manifest.py                            # 全部 spec
python scripts/validate_manifest.py --check-op SoftmaxFwdOp    # 单个算子，强制跑满五级
```

跑到第几级由 `status` 决定：`spec-only` 只跑 L0，`implemented` 跑满五级。

### 每个字段对不上时的表现 {#per-field}

spec 里的每个字段都有确定的消费者，也有确定的报错：

| 字段 | 谁读它 | 不一致时 |
| --- | --- | --- |
| `signature.inputs` / `outputs` | 算子层的校验、`_infer_output_shapes`、后端的 `build_kernel` 签名 | spec 校验器 L1 报错：参数名或顺序与 `forward` 不符 |
| `signature.shape_rules` | 算子层的形状校验 | spec 校验器 L2 报错：表达式不是合法的 Python，或与 `_infer_output_shapes` 的结论不一致 |
| `signature.dtype` 与 `dtype_combos` | 算子层的 dtype 校验 | spec 校验器 L3 报错：dtype 名不存在，或与 `_validate_dtypes` 不一致 |
| `workloads` | benchmark 的形状与 dtype | spec 校验器 L4 报错：benchmark 没有经 `load_workloads` 取形状 |
| `roofline` | `op.eval_roofline()`、性能报告里的效率 | 变量绑定不上，benchmark 直接报错 |
| `torch_compile_fullgraph` | CI 的 `compile-contract-gate` | 声明了却没有登记编译测试，这项检查失败 |
| `source` | spec 校验器定位 kernel、算子、测试与 benchmark 文件 | 路径不存在即报错 |

### 五级各查什么 {#five-levels}

| 级别 | 检查什么 |
| --- | --- |
| L0 | spec 的结构：必填字段在不在，各字段的类型对不对 |
| L1 | 签名的参数名与顺序，是否与 `forward` 一致 |
| L2 | `shape_rules` 的语法，以及它与 `_infer_output_shapes` 的结论是否一致 |
| L3 | dtype 名是否存在，以及它与 `_validate_dtypes` 是否一致 |
| L4 | benchmark 的形状是否经 `load_workloads` 取得 |

### 校验器不做什么 {#not-checked}

五级查的是语法与一致性，不是求值，三件事因此查不到：

- **`shape_rules` 不求值。** 校验器只看每条规则是不是合法的 Python 表达式，既不枚举调用的传法，也不区分「管传没传」与「管形状」两类规则。
- **传错的调用由算子自己拦。** 传了 `weight` 却不传 `bias`，spec 校验一样通过，报错来自 [`GroupNormFwdOp.forward`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/ops/norm/group_norm.py) 的运行时检查。
- **kernel 的执行细节不在 manifest 里。** 多 kernel 的执行顺序、累加 dtype、持久状态、tile 尺寸、autotune 配置，都不属于 spec 描述的内容。

完整的字段规范与全部规则见 [Op Manifest](design/manifest.md)。
