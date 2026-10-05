# 添加一个新 op

新增一个 op 需要在以下六个位置添加代码，表格的顺序也是推荐的编写顺序。

spec 最先编写，因为后面五个文件的内容都由 spec 决定，最后也都由 spec 校验。**spec 是这条流程的输入，其余五处都依照它编写。**{ .keystone }

| # | 文件 | 由谁对照 spec 检查 | 内容 |
| --- | --- | --- | --- |
| 1 | [`src/tileops/manifest/spec/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/manifest/spec)`<family>.yaml` | 校验器的 `schema` 与 `signature` 两级 | spec 本身 |
| 2 | [`src/tileops/ops/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/ops)`<family>/…` | 校验器对照 `__init__`、`forward` 与声明的 kernel；每次调用前后生成的检查 | op 类，继承 `Op` |
| 2 | [`src/tileops/ops/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/ops)`<family>/__init__.py` 与 [`src/tileops/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops)`<family>.py` | 校验器：family 的 `__all__` 与 manifest 一致 | op 名，由所属 family 导出，并出现在公开路径 `tileops.<family>.<Op>` 上 |
| 3 | [`src/tileops/kernels/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/kernels)`<family>/…` | —— | kernel 类，继承 `Kernel` |
| 4 | [`tests/ops/`](https://github.com/tile-ai/TileOPs/tree/main/tests/ops)`test_<名字>.py` | 契约测试，逐个运行每个 workload 行 | 与参考实现 `ref_program` 的数值比对 |
| 5 | [`benchmarks/ops/`](https://github.com/tile-ai/TileOPs/tree/main/benchmarks/ops)`bench_<名字>.py` | 校验器的 `bench` 级 | benchmark |

下文以最简单的矩阵乘 `GemmFwdOp` 为例，依次说明这六处。

## 第一步：写 spec

spec 各字段的含义与写法见[读写 manifest](user-guide/manifest/index.md)。新 op 的 spec 先写成 `status: spec-only`，表示接口已经确定、实现尚未完成。此时所有需要读取代码的检查都会跳过，不会因为找不到类而报错。

`GemmFwdOp` 的 spec 只列出一行 workload：

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
       dtype_cases: [{T: float16}, {T: bfloat16}], label: ds-v3-prefill-mlp-up}
  roofline:
    flops: "2 * M * N * K"
```

spec 不写文件路径，也不写 kernel。由哪些 kernel 服务这个 op 属于代码的范围，在第二步的 op 类上声明。

## 第二步：写 op 类 {#op-class}

op 类继承 [`Op`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/ops/op_base.py)，位于 spec 与 kernel 之间。每次调用前后的检查，包括 dtype、形状、约束与输出形状推导，都在类定义时依照签名生成，op 类不手写其中任何一项。op 类需要编写的是一次调用如何到达 kernel。

### 类的骨架与成员

`GemmFwdOp` 的骨架如下，略去 docstring：

```python
class GemmFwdOp(Op):
    compile_boundary: ClassVar[bool] = True           # optional: claims fullgraph=True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gemm_tma": GemmTMAKernel,
        "gemm_cp_async": GemmCpAsyncKernel,
        "gemv": GemvKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"gemm": GemmFwdInterface}

    def __init__(self, trans_a=False, trans_b=True, *, target=None, kernel_map=None, tune=False):
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)              # installs this instance's kernel map

    def forward(self, a, b):
        return self._call_boundary(a, b)              # the generated operator

    def _eager_forward(self, a, b):                   # the generated checks have run
        a, b = a.contiguous(), b.contiguous()         # handed over as the spec declares it
        m, k = (a.shape[1], a.shape[0]) if self.trans_a else a.shape
        call = GemmCall(                              # what this call is
            m=m,
            n=b.shape[0] if self.trans_b else b.shape[1],
            k=k,
            dtype=a.dtype,
            trans_a=self.trans_a,
            trans_b=self.trans_b,
            device=a.device,
        )
        return self.kernel_for("gemm", call)(a, b)
```

| # | 成员 | 编写依据 |
| --- | --- | --- |
| 1 | `__init__` | `signature.params` 的名字、顺序与默认值，再加上 `target`、`kernel_map`、`tune`；末尾调用 `self.dispatch_kernel(kernel_map)` |
| 2 | `kernel_types` | 能服务这个 op 的 Kernel 类，各对应一个 key；`kernel_map=` 按 key 替换其中一个 |
| 3 | `interfaces` | op 发出的每一个 kernel 调用各占一条，从 `kernel_for` 使用的名字映射到服务这个调用的各实现所继承的 `KernelInterface` 类 |
| 4 | `forward` | `signature.inputs` 的顺序，可选输入排在最后，默认值为 `None` |
| 5 | `_eager_forward` | 把输入转为连续张量，构造 call spec，取得 kernel，再调用它 |
| 6 | `compute_roof` | 可选。表示 op 的 FLOPs 按哪个硬件单元的峰值计算，默认是 CUDA core 上的 fp32，只在使用其他单元时编写 |

`_infer_output_shapes`、`_validate_dtypes` 与 `eval_roofline` 都依照 spec 生成，不需要编写。

不声明编译边界的 op 把 `_eager_forward` 的内容直接写在 `forward` 中。声明了编译边界的 op 把这些内容放在生成的 operator 之后执行。具体做法见[接入 torch.compile](torch-compile.md)。

### `kernel_for` 与 kernel 的选择 {#kernel-selection}

kernel 是编译产物，构造一次需要几百毫秒到几秒，而一个 op 实例会以不同的形状与 dtype 被反复调用。因此 op 层维护一张缓存表：本次调用所需的 kernel 已经构造过时直接取出，否则构造后存入表中。`kernel_for` 是 in-tree 实现访问这张表的唯一入口；[target](backends.md) 服务的是整个 op，不经过 `kernel_for`。

`kernel_for` 接受两个参数：

- **`interface`**：`interfaces` 的一个 key，指 op 发出的某一个 kernel 调用。`GemmFwdOp` 只发出一个 kernel 调用，因此只声明一个 `"gemm"`。只有语义或调用契约改变时才新增一个 kernel 接口，例如 `BatchNormFwdOp` 的 `batch_norm_fwd_train` 与 `batch_norm_fwd_infer` 返回的内容不同。在某个形状范围或某个架构上更快的 kernel 是已有 kernel 接口的另一个实现。
- **`call`**：一个冻结的 `CallSpec` 子类，包含选择实现和构造 kernel 所需的信息，即形状、dtype、op 的语义参数与设备。它必须是这个 kernel 接口的 `request` 指定的类型。设备事实（`arch`、`sm_count`、`calibration`、`smem_budget`）由派发机制在缓存未命中时从 `call.device` 推出，调用方不填写。

op 按 kernel 接口的抽象 `forward` 方法所声明的参数及其顺序调用取得的 kernel。

kernel 接口是写在 [`src/tileops/kernels/<family>/call_spec.py`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/kernels) 中的一个类，与它在 `request` 中指定的 call spec 放在一起；只有一个 kernel 文件的 family 把两者都写在那个文件中。类名的格式是 `{Name}{Fwd|Bwd}Interface`，表示变体的词写在方向之前，由 `interface-names-lint` 检查。kernel 接口的抽象 `forward` 是所有实现（in-tree 实现与 backend 提供的实现）共同依据的唯一契约，因此它的 docstring 写明每个张量的形状、dtype、内存布局、设备，以及是否被原地写入：

```python
class GemmFwdInterface(KernelInterface):
    """Dense matmul under the ``(trans_a, trans_b)`` layout the call states."""

    request = GemmCall

    @abstractmethod
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiply the two matrices; nothing is written in place.

        Both operands are contiguous on ``call.device`` in ``call.dtype``, which is
        ``float16`` or ``bfloat16``; the contraction accumulates in ``float32``.

        Args:
            a: ``(call.m, call.k)``, or ``(call.k, call.m)`` when ``call.trans_a``.
            b: ``(call.n, call.k)`` when ``call.trans_b``, else ``(call.k, call.n)``.

        Returns:
            A new ``(call.m, call.n)`` tensor in ``call.dtype``.
        """
```

实现是同时继承 `Kernel` 与某一个 kernel 接口的类，以一个 key 列在 `kernel_types` 中。一次调用由哪个实现服务，取决于各实现自己声明的可用性、适用范围与优先关系，op 不参与选择。只有一个实现的 kernel 接口，除继承 kernel 接口外不需要其他声明。选择规则、各项声明的写法与常见报错见 [op 如何选择 kernel](user-guide/dispatch/index.md) 与[如何为 op 新增 kernel](user-guide/dispatch/writing.md)。

op 不编写 `entry_for`，也不自行维护 kernel 缓存。完全没有 in-tree 实现、只依赖外部 backend 的 op 不写 `kernel_types` 与 `interfaces`；没有 target 认领调用设备时，调用抛出 `OpNotAvailableError`。backend 新增实现或替换某个 key 的方式见 [backend 如何接入](user-guide/dispatch/backends.md)。

### 注册

op 名需要加入两处的导入与 `__all__`：

1. 所属 family 的 [`src/tileops/ops/<family>/__init__.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/ops)，即类的实现位置；
1. [`src/tileops/<family>.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops)，即公开路径。

缺少后者时，`from tileops.<family> import ...` 无法导入这个 op，API 参考也不会收录它。

## 第三步：写 kernel

kernel 类继承 [`Kernel`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/kernels/kernel_base.py) 与它实现的 kernel 接口，放在 [`src/tileops/kernels/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/kernels) 下，用 TileLang 编写，在构造时编译。kernel 类实现 `forward`，由基类的 `__call__` 调用。构造函数由本类 `entry_for` 返回的 builder 调用；`forward` 接受 kernel 接口规定的参数，也就是第二步中的 `kernel(a, b)`。

kernel 是这六处中唯一不受 spec 约束的一处：kernel 不读取 spec，也不对照 spec 检查。

构造参数与调用参数的划分有一条硬性要求：**只有会被编译进生成代码的值才作为构造参数。** `GemmTMAKernel` 的划分如下：

```python
class GemmTMAKernel(Kernel, GemmFwdInterface):
    def __init__(self, m, n, k, dtype, config=None, tune=False, trans_a=False, trans_b=False, ...):
        self.kernel = _gemm_kernel(m, n, k, trans_a, trans_b, self.dtype_str, ...)  # compiles
        self.init_config(config, tune)      # tile sizes and pipeline depth

    def forward(self, a, b):                # a call passes tensors, nothing else
        ...
```

`m`、`n`、`k`、dtype 与两个布局标志是构造参数，因为它们在生成的代码中是常量：循环边界、TMA 描述符与 WGMMA 的形状都按这些值展开，tile 尺寸也是如此。张量本身由 `forward` 接收，每次调用只更换指针。

划分错误的后果是重新编译。decode 逐步推进时，`seq_len` 每一步加 1，batch 随 running set 变化：

```python
# 错：seq_len 进了构造函数 —— 每一步都是一个新 kernel
kernel = AttnKernel(batch, seq_len, num_heads, dtype)

# 对：只有编译期常量进构造函数，变化的量随调用传入
kernel = AttnKernel(num_heads, head_dim, dtype)
out = kernel(q, k, v)                       # seq_len 从张量形状里读
```

在前一种写法下，`entry_for` 返回的 build identity 包含 `seq_len`，每一步都无法命中缓存，每一步都编译一次，decode 因此无法正常运行。

## 第四步：写测试

测试放在 [`tests/ops/`](https://github.com/tile-ai/TileOPs/tree/main/tests/ops) 中，比对对象是 workload（或测试类）定义的参考实现 `ref_program`。测试自行选择形状，以覆盖 kernel 的各个分支。用例按运行时机分为三类：

- `smoke` 用例在每个 PR 中运行；
- `full` 用例在修改了其测试文件的 PR 和 nightly 中运行；
- 耗时长的用例标记为 `nightly`，只在 nightly 中运行。

workload 行不属于单元测试的覆盖范围，因为契约测试已经用 op 运行过每一行。

测试骨架使用 [`tests/workload_test_base.py`](https://github.com/tile-ai/TileOPs/blob/main/tests/workload_test_base.py) 中的 `TestBase`，用例写在 `PARAMS` 中。

op 有可选输入时，传入与不传入各至少需要一条用例，因为两种情况通常走不同的 kernel。

## 第五步：写 benchmark

benchmark 放在 [`benchmarks/ops/`](https://github.com/tile-ai/TileOPs/tree/main/benchmarks/ops) 中。每个调用由一个基于 op 与该调用的 workload 构造的 `ManifestBenchmark` 计时。调用不手写：`manifest_calls(<Op>)` 为每个 workload 行的每个 dtype case 各实例化一个调用，并以 case id 命名。手写调用的 benchmark 无法通过校验器的 `bench` 级：

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

benchmark 至少还需要记录一个非 TileOPs 的基线，否则这一行没有比较对象。基线需要转换输入时，转换代码保留在基线自己的计时区间内。报出的各个数字的含义见 [benchmark 的计时方法](timing.md)。

## 第六步：反转实现状态，让 op 进入 CI 校验

上述五处都写完之后，先运行以下三条命令自查：

```bash
python scripts/validate_manifest.py --check-op GemmFwdOp   # spec and code agree
python -m pytest tests/ops/test_gemm.py -v                # numerics match ref_program
python -m pytest benchmarks/ops/bench_gemm.py             # the benchmark produces numbers
```

三条命令都通过后，再把 spec 的 `status` 从 `spec-only` 改为 `implemented`。这一改动启用所有需要读取代码的检查，op 由此进入 CI 的保护范围：此后每次改动，spec 校验器、测试与 nightly benchmark 都会对照 spec 检查一遍。

## 后续步骤

op 能够运行之后，还有两项可选的工作：

- 让 op 能够进入使用者的编译图，见[接入 torch.compile](torch-compile.md)。
- 让 op 在其他硬件上由其他 kernel 服务，见[接入新硬件 backend](backends.md)。
