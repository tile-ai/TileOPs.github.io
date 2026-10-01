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
       dtype_cases: [{T: float16}, {T: bfloat16}], label: ds-v3-prefill-mlp-up}
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
        "gemm_tma": GemmTmaKernel,
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

| # | 成员 | 照什么写 |
| --- | --- | --- |
| 1 | `__init__` | `signature.params` 的名字、顺序与默认值，再加 `target`、`kernel_map`、`tune`；结尾调用 `self.dispatch_kernel(kernel_map)` |
| 2 | `kernel_types` | 能服务这个算子的 Kernel 类，各起一个 key；`kernel_map=` 按这个 key 替换其中一个 |
| 3 | `interfaces` | 算子发出的每一个 kernel 调用各占一条：`kernel_for` 用的名字 → 服务这个调用的各实现所继承的 `KernelInterface` 类 |
| 4 | `forward` | `signature.inputs` 的顺序，可选输入排在最后、默认 `None` |
| 5 | `_eager_forward` | 把输入变成连续的，构造 call spec，取出 kernel，再调用它 |
| 6 | `compute_roof` | 可选。算子的 FLOPs 按哪个硬件单元的峰值定价，默认是 CUDA core 上的 fp32，用别的单元时才写 |

`_infer_output_shapes`、`_validate_dtypes` 与 `eval_roofline` 都照 spec 生成，不用写。

不声明编译边界的算子，把 `_eager_forward` 的内容直接写在 `forward` 里；声明了边界，这些内容挪到生成的 operator 后面。做法见[接入 torch.compile](torch-compile.md)。

### `kernel_for` 与 kernel 的选择 {#kernel-selection}

kernel 是编译产物，构造一次要几百毫秒到几秒，而一个算子实例会被反复调用，形状与 dtype 各不相同。算子层因此维护一张记忆表：本次调用要的 kernel 已经构造过就取回来，没有才构造并存进去。`kernel_for` 是自带实现走到这张表的唯一入口；[target](backends.md) 服务的是整个算子，不经过它。

`kernel_for` 接受两个参数：

- **`interface`**：`interfaces` 的一个 key，指算子发出的某一个 kernel 调用。`GemmFwdOp` 只发出一个 kernel 调用，因此只声明一个 `"gemm"`。只有语义或调用契约改变时才新开一个接口：`BatchNormFwdOp` 的 `batch_norm_fwd_train` 与 `batch_norm_fwd_infer` 返回的东西不同。某个形状范围或某个架构上更快的 kernel 是已有接口的另一个实现。
- **`call`**：一个冻结的 `CallSpec` 子类，带着选择实现和构造 kernel 所需的信息：形状、dtype、算子的语义参数与设备。它必须是这个接口 `request` 指定的类型。设备事实（`arch`、`sm_count`、`calibration`、`smem_budget`）由派发机制在未命中时从 `call.device` 推出，调用方不填。

算子按接口的抽象 `forward` 方法所声明的参数、以同样的顺序调用取回的 kernel。

接口是写在 [`src/tileops/kernels/<family>/call_spec.py`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/kernels) 里的一个类，与它在 `request` 中指名的 call spec 放在一起；只有一个 kernel 文件的 family 把两者都写在那个文件里。名字是 `{Name}{Fwd|Bwd}Interface`，变体词写在方向之前，由 `interface-names-lint` 检查。它的抽象 `forward` 是各实现（自带的与后端提供的）唯一依据的契约，docstring 因此写明每个张量的形状、dtype、内存布局、设备，以及是否被原地写入：

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

实现是同时继承 `Kernel` 与某一个接口的类，以一个 key 列在 `kernel_types` 中。一次调用由哪个实现服务，取决于各实现自己的四项声明，算子不参与：

| # | 声明 | 说明什么 | 未声明时的默认 |
| --- | --- | --- | --- |
| 1 | `devices`、`supported_archs` | 实现能在哪些设备上运行 | CUDA 设备，全部架构 |
| 2 | `applies(call)`、`refusal(call)` | 实现服务哪些调用，正面写出 | 服务全部调用 |
| 3 | `general`、`preferred_over` | 两个实现都服务同一次调用时谁胜出 | 不胜过任何实现 |
| 4 | `entry_for(call)` | build identity，以及每个 identity 只运行一次的 builder | 以整个 call spec 为 identity，用 `cls(call)` 构造 |

派发机制先按可用性过滤，再在剩下的、且适用的实现中选出唯一的胜者：`general` 的实现低于其他所有实现，其余按各自 `preferred_over` 列出的 key 比较。一个实现都不剩时报 `no implementation serves this call`；没有任何 key 能在这次调用的设备类型上运行时报 `OpNotAvailableError`；剩下的实现之间互相没有优先关系时报 `dispatch is ambiguous`。实现的声明顺序不影响选择结果。需要让出一段范围时，由应当胜出的一方声明 `preferred_over`，而不是让另一方在自己的 `applies` 里把这段范围排除掉。

`GemmFwdOp` 的三个实现这样分担 `"gemm"` 接口的调用：

| # | key | 服务 | 声明 |
| --- | --- | --- | --- |
| 1 | `gemm_tma` | 操作数能被 TMA 寻址的 SM90 形状 | `supported_archs = [90]`，以及给出未对齐原因的 `refusal` |
| 2 | `gemv` | 沿 K 规约且最多两行的形状，这种形状在 CUDA core 上规约更快 | `supported_archs = [90]`、经 `band_for` 实现的 `applies`、`preferred_over = frozenset({"gemm_tma"})` |
| 3 | `gemm_cp_async` | 其余两个都不认领的全部形状，前提是一行 K 至少占满一次 4 字节读取 | `supported_archs = [80, 86, 89, 90]`、`general = True`，以及拒绝更窄 K 行的 `refusal` |

`entry_for(call)` 返回两样东西：两次调用共享它才算同一个 kernel 的 **build identity**，以及每个 identity 只运行一次的 **builder**。identity 少带一个量，第二种 dtype 就会复用第一种 dtype 的 kernel；kernel 只依赖其中几个量却把整个形状都带上，就变成每个形状各编译一次。

只有一个实现的接口，除继承接口外不需要别的声明。`RMSNormKernel` 是 `RMSNormFwdOp` 唯一的实现（[`src/tileops/kernels/norm/rms_norm.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/kernels/norm/rms_norm.py)）：

```python
class RMSNormKernel(Kernel, RMSNormFwdInterface):
    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def entry_for(cls, call: LayerNormCall) -> Entry:
        identity = (call.n, call.eps, call.dtype)
        return identity, lambda: cls(*identity)
```

算子自己不写 `entry_for`，也不自建 kernel 缓存：没有缓存字典，也不以某个属性是否已赋值来决定要不要构造。把 `kernel_for` 的返回值存进 `self.kernel` 不算自建缓存。

完全没有自带实现、只依赖外部后端的算子，`kernel_types` 与 `interfaces` 都不写；在没有 target 认领设备时，调用会抛 `OpNotAvailableError`。

后端可以为一个接口新增实现，也可以替换某个 key 登记的类，这两件事都不改动 TileOPs，见[接入一类新硬件](backends.md)。

### 注册

把算子名加进两处的导入与 `__all__`：算子所属家族的 [`src/tileops/ops/<family>/__init__.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/ops)（类的实现位置），以及 [`src/tileops/<family>.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops)（公开路径）。缺了后者，`from tileops.<family> import ...` 拿不到这个算子，API 参考也收不到它。

## 第三步：写 kernel

kernel 类继承 [`Kernel`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/kernels/kernel_base.py) 与它实现的那个接口，放在 [`src/tileops/kernels/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/kernels) 下，用 TileLang 编写，在构造时编译。它实现 `forward`，基类的 `__call__` 会调用它。构造函数由本类 `entry_for` 返回的 builder 调用，`forward` 接受接口规定的参数，就是第二步里的 `kernel(a, b)`。

它是这六处里唯一不受 spec 约束的一处：kernel 不读 spec，也不对照 spec 检查。

构造参数与调用参数的划分有一条硬性要求：**只有会被编译进生成代码的值才进构造函数。** `GemmTmaKernel` 是这样分的：

```python
class GemmTmaKernel(Kernel, GemmFwdInterface):
    def __init__(self, m, n, k, dtype, config=None, tune=False, trans_a=False, trans_b=False, ...):
        self.kernel = _gemm_kernel(m, n, k, trans_a, trans_b, self.dtype_str, ...)  # compiles
        self.init_config(config, tune)      # tile sizes and pipeline depth

    def forward(self, a, b):                # a call passes tensors, nothing else
        ...
```

`m`、`n`、`k`、dtype 与两个布局标志进了构造函数，因为生成的代码里这些值是常量：循环边界、TMA 描述符、WGMMA 的形状都按它们展开，tile 尺寸同理。张量本身留给 `forward`，每次调用只换指针。

分错的代价是重新编译。decode 一步一步往前走，`seq_len` 每步 +1，batch 随 running set 变化：

```python
# 错：seq_len 进了构造函数 —— 每一步都是一个新 kernel
kernel = AttnKernel(batch, seq_len, num_heads, dtype)

# 对：只有编译期常量进构造函数，变化的量随调用传入
kernel = AttnKernel(num_heads, head_dim, dtype)
out = kernel(q, k, v)                       # seq_len 从张量形状里读
```

上一种写法下，`entry_for` 返回的 build identity 里带着 `seq_len`，每步都未命中、每步都编译一次，decode 直接跑不动。

## 第四步：写测试

测试放在 [`tests/ops/`](https://github.com/tile-ai/TileOPs/tree/main/tests/ops)，比对对象是 workload（或测试类）定义的参考实现 `ref_program`，形状由测试自己挑，以覆盖 kernel 的各个分支；`smoke` 用例每个 PR 都运行；`full` 用例在改动其测试文件的 PR 和 nightly 中运行；耗时长的用例标 `nightly`，只在 nightly 运行。workload 行不是单元测试的覆盖面，契约测试已经把每一行都交给算子跑过。

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
