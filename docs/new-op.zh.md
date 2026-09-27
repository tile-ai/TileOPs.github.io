# 添加一个新算子

写一个新算子需要在以下六个位置添加实现，表格的顺序也是推荐的动手顺序。

其中 spec 要第一个写：后面五个文件的内容都由它决定，最后也都由它校验。**spec 是这条流程的输入，其余五处都是照它写出来的。**{ .keystone }

| # | 文件 | 由谁对照 spec 检查 | 内容 |
| --- | --- | --- | --- |
| 1 | [`src/tileops/manifest/spec/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/manifest/spec)`<family>.yaml` | 校验器的 `schema` 与 `signature` 两级 | spec 本身 |
| 2 | [`src/tileops/ops/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/ops)`<family>/…` | 校验器对照 `__init__`、`forward` 与声明的 kernel；每次调用前后生成的检查 | 算子类，继承 `Op` |
| 2 | [`src/tileops/ops/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/ops)`<family>/__init__.py` 与 [`src/tileops/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops)`<family>.py` | 校验器：家族的 `__all__` 与 manifest 一致 | 算子名，由所属家族导出，并出现在公开路径 `tileops.<family>.<Op>` 上 |
| 3 | [`src/tileops/kernels/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/kernels)`<family>/…` | —— | kernel 类，继承 `Kernel` |
| 4 | [`tests/ops/`](https://github.com/tile-ai/TileOPs/tree/main/tests/ops)`test_<名字>.py` | 契约测试，逐个跑每个 workload 行 | 与参考实现 `ref_program` 的数值比对 |
| 5 | [`benchmarks/ops/`](https://github.com/tile-ai/TileOPs/tree/main/benchmarks/ops)`bench_<名字>.py` | 校验器的 `bench` 级 | benchmark |

下文以最简单的矩阵乘 `GemmFwdOp` 为例走一遍这六处。

## 第一步：写 spec

spec 各字段的含义与写法见[读写 manifest](manifest.md)。新算子先写成 `status: spec-only`，表示接口已经定下来、实现还没有，这时需要读代码的检查都跳过，不会因为找不到类而报错。

`GemmFwdOp` 的 spec，workload 只列一行：

```yaml
GemmFwdOp:
  ref_api: torch.matmul
  family: gemm
  status: spec-only
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
    - {M: 4096, N: 4096, K: 7168, trans_a: false, trans_b: true,
       dtype_cases: [{T: float16}, {T: bfloat16}], label: ds-v3-prefill-attn-proj}
  roofline:
    flops: "2 * M * N * K"
```

spec 不写文件路径，也不写 kernel：由哪些 kernel 服务这个算子是代码里的事，在第二步的算子类上声明。

## 第二步：写算子类 {#op-class}

算子类继承 [`Op`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/ops/op_base.py)，是 spec 与 kernel 之间的一层。每次调用前后的检查 —— dtype、形状、约束、输出形状推导 —— 都在类定义时照签名生成，算子类一条也不写。它要写的是一次调用怎么走到 kernel。

### 类的骨架与成员

`GemmFwdOp` 的骨架，略去 docstring：

```python
class GemmFwdOp(Op):
    compile_boundary: ClassVar[bool] = True           # optional: claims fullgraph=True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gemm_tma_kernel": GemmTmaKernel,
        "gemm_cp_async_kernel": GemmCpAsyncKernel,
        "gemv_kernel": GemvKernel,
    }

    def __init__(self, trans_a=False, trans_b=True, *, target=None, kernel_map=None, tune=False):
        self.trans_a, self.trans_b = trans_a, trans_b
        self.target, self.tune = target, tune
        self.dispatch_kernel(kernel_map)              # installs this instance's kernel map

    def forward(self, a, b):
        return self._call_boundary(a, b)              # the generated operator

    def _eager_forward(self, a, b):                   # the generated checks have run
        a, b = a.contiguous(), b.contiguous()         # handed over as the spec declares it
        m, k = (a.shape[1], a.shape[0]) if self.trans_a else a.shape
        n = b.shape[0] if self.trans_b else b.shape[1]
        kernel = self.kernel_for(
            "gemm",                                   # the memoization bucket
            (a, b),                                   # the tensors the kernel gets
            self._call_spec(m, n, k, a.dtype, a.device),  # what this call is
        )
        return kernel(a, b)
```

| # | 成员 | 照什么写 |
| --- | --- | --- |
| 1 | `__init__` | `signature.params` 的名字、顺序与默认值，再加 `target`、`kernel_map`、`tune`；结尾调用 `self.dispatch_kernel(kernel_map)` |
| 2 | `kernel_types` | 能服务这个算子的 Kernel 类，各起一个名字；`kernel_map=` 按这个名字替换其中一个 |
| 3 | `forward` | `signature.inputs` 的顺序，可选输入排在最后、默认 `None` |
| 4 | `_eager_forward` | 连续化、调用记录、取 kernel、launch kernel |
| 5 | `compute_roof` | 可选：给算子 FLOPs 定价的 GPU profile 单元，不是 CUDA core fp32 时才写 |

`_infer_output_shapes`、`_validate_dtypes` 与 `eval_roofline` 都照 spec 生成，不用写。

不声明编译边界的算子，把 `_eager_forward` 的内容直接写在 `forward` 里；声明了边界，这些内容挪到生成的 operator 后面。做法见[接入 torch.compile](torch-compile.md)。

### `kernel_for` 与 kernel 的选择

kernel 是编译产物，构造一次要几百毫秒到几秒，而一个算子实例会被反复调用，形状与 dtype 各不相同。算子层因此维护一张记忆表：本次调用要的 kernel 已经构造过就取回来，没有才构造并存进去。`kernel_for` 是自带实现走到这张表的唯一入口；[target](backends.md) 服务的是整个算子，不经过它。

三个参数：

- **`role`**：记忆表的桶，算子一次调用跑几个 kernel 就有几个。`GemmFwdOp` 只跑一个，所以只有一个 role，不论三个类中哪一个服务这次调用。
- **`inputs`**：即将传给 kernel 的张量，顺序照 `signature.inputs`，一个输入占一个位置。没传的可选输入留下位置，值为 `None`。
- **`call`**：本次调用是什么。`GemmCall` 带着 GEMM 各 kernel 要读的全部事实：`m`、`n`、`k`、dtype、布局、设备。

由哪个类服务一次调用，由这些类自己决定，不由算子决定。每个类声明自己服务的范围（`applies`、`refusal`），其中一个标为 `general`，负责其余情形；两个专用类同时认领一次调用会报错，不会静默挑一个。选中的类用 `entry_for(call)` 返回两样东西：两次调用要共享什么才算同一个 kernel 的**身份**，以及每个身份只跑一次的**构造方法**。身份带少了，第二种 dtype 会复用第一种 dtype 的 kernel；kernel 只依赖其中几个量却把整个形状带上，就变成一个形状编译一次。

只有一个 kernel、也没有调用记录的算子，在算子类上自己写 `entry_for(role, call)`，在那里给出身份与构造方法，`RMSNormFwdOp` 就是这样：

```python
def entry_for(self, role, call):                    # call is the input dtype
    n = math.prod(self.normalized_shape)
    eps = torch.finfo(torch.float32).eps if self.eps is None else float(self.eps)
    return call, lambda: self.kernel_map["rms_norm"](n, eps, call, tune=self.tune)
```

完全没有自带实现、只依赖外部后端的算子，`kernel_types` 与 `entry_for` 都不写；在没有 target 认领设备时，调用会抛 `OpNotAvailableError`。

### 注册

把算子名加进两处的导入与 `__all__`：算子所属家族的 [`src/tileops/ops/<family>/__init__.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/ops)（类的实现位置），以及 [`src/tileops/<family>.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops)（公开路径）。缺了后者，`from tileops.<family> import ...` 拿不到这个算子，API 参考也收不到它。

## 第三步：写 kernel

kernel 类继承 [`Kernel`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/kernels/kernel_base.py)，放在 [`src/tileops/kernels/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/kernels) 下，用 TileLang 写，构造时编译、`__call__` 时启动。构造函数由它的 `entry_for` 构造方法调用，调用签名就是第二步里的 `kernel(a, b)`。

它是这六处里唯一不受 spec 约束的一处：kernel 不读 spec，也不对照 spec 检查。

构造参数与调用参数的划分有一条硬性要求：**只有会被编译进生成代码的值才进构造函数。** `GemmTmaKernel` 是这样分的：

```python
class GemmTmaKernel(Kernel):
    def __init__(self, m, n, k, dtype, config=None, tune=False, trans_a=False, trans_b=False, ...):
        self.kernel = _gemm_kernel(m, n, k, trans_a, trans_b, self.dtype_str, ...)  # compiles
        self.init_config(config, tune)      # tile sizes and pipeline depth

    def __call__(self, a, b):               # a call passes tensors, nothing else
        ...
```

`m`、`n`、`k`、dtype 与两个布局标志进了构造函数，因为生成的代码里这些值是常量：循环边界、TMA 描述符、WGMMA 的形状都按它们展开，tile 尺寸同理。张量本身留给 `__call__`，每次调用只换指针。

分错的代价是重新编译。decode 一步一步往前走，`seq_len` 每步 +1，batch 随 running set 变化：

```python
# 错：seq_len 进了构造函数 —— 每一步都是一个新 kernel
kernel = AttnKernel(batch, seq_len, num_heads, dtype)

# 对：只有编译期常量进构造函数，变化的量随调用传入
kernel = AttnKernel(num_heads, head_dim, dtype)
out = kernel(q, k, v)                       # seq_len 从张量形状里读
```

上一种写法下，`entry_for` 返回的身份里带着 `seq_len`，每步都未命中、每步都编译一次，decode 直接跑不动。

## 第四步：写测试

测试放在 [`tests/ops/`](https://github.com/tile-ai/TileOPs/tree/main/tests/ops)，比对对象是 workload（或测试类）定义的参考实现 `ref_program`，形状由测试自己挑，以覆盖 kernel 的各个分支；小形状标 `smoke` 进 PR 检查，大形状标 `full` 留给 nightly。workload 行不是单元测试的覆盖面，契约测试已经把每一行都交给算子跑过。

骨架用 [`tests/test_base.py`](https://github.com/tile-ai/TileOPs/blob/main/tests/test_base.py) 里的 `TestBase` 与 `FixtureBase`，用例写在 `PARAMS` 里。

如果这个算子有可选输入，传与不传各至少要有一条用例 —— 两侧走的往往是不同的 kernel。

## 第五步：写 benchmark

benchmark 放在 [`benchmarks/ops/`](https://github.com/tile-ai/TileOPs/tree/main/benchmarks/ops)，每个调用交给一个围绕算子与该调用 workload 构造的 `ManifestBenchmark` 计时。调用不自己写：`manifest_calls(<Op>)` 把每个 workload 行配上它的每个 dtype case 各实例化一次，并以 case id 命名；自己写调用的 benchmark 过不了校验器的 `bench` 级：

```python
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops import GemmFwdOp
from workloads.gemm import GemmWorkload


@pytest.mark.parametrize("call", manifest_calls(GemmFwdOp))
def test_gemm_bench(call) -> None:
    workload = GemmWorkload.from_call(call)
    a, b = workload.gen_inputs()
    op = GemmFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, workload)
    bm.compare({"tileops": op, "torch-cublas": workload.ref_program}, a, b)
```

另外至少要记一个非 TileOPs 的基线，否则这一行没有比较对象。基线若需要转换输入，转换的代码留在它自己的计时区间内，不要挪出去。报出来的数字各是什么意思，见[benchmark 怎么计时](timing.md)。

## 第六步：反转实现状态，让算子进入 CI 校验

上面五处都写完之后，先跑下面三条命令自查一遍：

```bash
python scripts/validate_manifest.py --check-op GemmFwdOp   # spec and code agree
python -m pytest tests/ops/test_gemm.py -v                # numerics match ref_program
python -m pytest benchmarks/ops/bench_gemm.py             # the benchmark produces numbers
```

三样都过，再把 spec 的 `status` 从 `spec-only` 反转成 `implemented`。这一改动打开所有需要读代码的检查，算子由此进入 CI 的保护范围：往后每次改动，spec 校验器、测试与 nightly benchmark 都会对照 spec 检查一遍。

## 接下来

算子跑起来之后，还有两件可选的事：

- 让算子能进使用者的编译图 —— [接入 torch.compile](torch-compile.md)。
- 让它在别的硬件上由别人的 kernel 服务 —— [接入新硬件后端](backends.md)。
