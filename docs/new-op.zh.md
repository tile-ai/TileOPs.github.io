# 添加一个新算子

写一个新算子需要在以下六个位置添加实现，表格的顺序也是推荐的动手顺序。

其中 spec 要第一个写：后面五个文件的内容都由它决定，最后也都由它校验。**spec 是这条流程的输入，其余五处都是照它写出来的。**{ .keystone }

| # | 文件 | spec 里由谁指名 | 内容 |
| --- | --- | --- | --- |
| 1 | [`src/tileops/manifest/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/manifest)`<family>.yaml` | 顶层的名字就是算子类名 | spec 本身 |
| 2 | [`src/tileops/ops/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/ops)`<family>/…` | `source.op` | 算子类，继承 `Op` |
| 2 | [`src/tileops/ops/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/ops)`<family>/__init__.py` 与 [`src/tileops/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops)`<family>.py` | —— | 算子名，由所属家族导出，并出现在公开路径 `tileops.<family>.<Op>` 上 |
| 3 | [`src/tileops/kernels/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/kernels)`<family>/…` | `source.kernel` | kernel 类，继承 `Kernel` |
| 4 | [`tests/ops/`](https://github.com/tile-ai/TileOPs/tree/main/tests/ops)`test_<名字>.py` | `source.test` | 与 `ref_api` 的数值比对 |
| 5 | [`benchmarks/ops/`](https://github.com/tile-ai/TileOPs/tree/main/benchmarks/ops)`bench_<名字>.py` | `source.bench` | benchmark |

下文以最简单的矩阵乘 `GemmFwdOp` 为例走一遍这六处。

## 第一步：写 spec

spec 各字段的含义与写法见[读写 manifest](manifest.md)，这里只说新算子特有的两件事。

第一件是状态。新算子先写成 `status: spec-only`，表示接口已经定下来、实现还没有，这时校验只跑 L0（结构检查），不会因为找不到实现而报错。

第二件是 `source.kernel_map`，这份 spec 里唯一推导不出来的字段。

一个算子可能对应多个 kernel：GEMM 在一般形状下用矩阵乘的 kernel，M 为 1 时退化成矩阵向量乘，换另一个 kernel 更快。`kernel_map` 就是这几个 kernel 的名单，每个 kernel 起一个名字，对应它的 Kernel 类：

```yaml
GemmFwdOp:
  ref_api: torch.matmul
  family: gemm
  status: spec-only
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
  source:
    kernel: tileops/kernels/gemm/dense.py
    kernel_map:
      gemm_kernel: GemmKernel
      gemv_kernel: GemvKernel
    op: tileops/ops/gemm/gemm.py
    test: tests/ops/test_gemm.py
    bench: benchmarks/ops/bench_gemm.py
```

算子在运行时按这些名字取 kernel：`_eager_forward` 里挑出用哪一个，把名字传给 `get_or_build_kernel`，算子层再从 `kernel_map` 找到对应的类去构造（见[第二步](#op-class)）。外部后端也是照这份名单注册的 —— 它为哪个名字注册 `build_kernel`，就接管了算子的哪一个 kernel。

名字自己起，但要和 kernel 的用途对得上，而且写进算子代码之后就不该再改：它同时是 spec、算子实现与外部后端三方约定的那个词。这也是它推导不出来的原因 —— 只有写 kernel 的人知道这个算子要分几种情形。

## 第二步：写算子类 {#op-class}

算子类继承 [`Op`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/ops/op_base.py)，是 spec 与 kernel 之间的一层：它按 spec 校验入参、推导输出形状，再取到 kernel 并 launch。先写它，是因为它的内容全部由 spec 决定，而它调用 kernel 的那一行同时定下了 kernel 的构造签名。

### 类的骨架与四个成员

`GemmFwdOp` 的骨架，方法体略去与本页无关的部分：

```python
class GemmFwdOp(Op):
    def __init__(self, trans_a=False, trans_b=True, kernel_map=None, tune=False):
        self.trans_a, self.trans_b, self.tune = trans_a, trans_b, tune
        self.dispatch_kernel(kernel_map)             # 建立 kernel_map，不可省略

    @property
    def default_kernel_map(self):                    # spec 的 source.kernel_map
        return {"gemm_kernel": GemmKernel, "gemv_kernel": GemvKernel}

    def _infer_output_shapes(self, a_shape, b_shape):
        m = a_shape[1] if self.trans_a else a_shape[0]
        n = b_shape[0] if self.trans_b else b_shape[1]
        return {"d": (m, n)}                         # spec 的 shape_rules

    def forward(self, a, b):
        self._validate_dtypes(a, b)                  # 基类按 spec 生成，直接调用
        m, n, k = self._infer_mnk(a, b)
        a, b = a.contiguous(), b.contiguous()        # 按 spec 声明的形状交给 kernel
        slot = "gemv_kernel" if m == 1 else "gemm_kernel"
        kernel = self.get_or_build_kernel(
            slot,                                    # kernel_map 里的名字
            (a, b),                                  # 外部路径按这些张量查表，后端也收到它们
            key=(m, n, k, a.dtype),                  # 自带 kernel 按什么查表
            build=lambda: self.kernel_map[slot](m, n, k, a.dtype, tune=self.tune),
        )
        return kernel(a, b)
```

要自己写的是这四个成员，内容都从 spec 来：

| # | 成员 | 照 spec 的哪一部分写 |
| --- | --- | --- |
| 1 | `__init__` | `signature.params` 的名字与默认值，另加 `kernel_map` 与 `tune`；结尾调用 `self.dispatch_kernel(kernel_map)` 建立本实例的 kernel_map |
| 2 | `default_kernel_map` | `source.kernel_map`：名字照抄，取值换成 Kernel 类本身 |
| 3 | `_infer_output_shapes` | `signature.shape_rules` 里推导输出形状那几条 |
| 4 | `forward` | `signature.inputs` 的顺序与默认值（可选输入排在必填之后），加上校验、连续化、取 kernel、launch kernel |

另有两个成员不用写：`_validate_dtypes` 与 `eval_roofline` 由基类在子类定义时照 spec 的 dtype 声明与 `roofline` 生成并装上，直接调用即可，只有需要特殊行为时才自己覆写。

### `get_or_build_kernel`

kernel 是编译产物，构造一次要几百毫秒到几秒，而一个算子实例会被反复调用，形状与 dtype 各不相同。算子层因此维护一张记忆表：本次调用要的 kernel 已经构造过就取回来，没有才构造并存进去。`get_or_build_kernel` 是这张表唯一的入口，也是自带实现与外部后端的分岔点（[后端协议](backends.md)里的第二层选择）。

四个参数：

**`name`** —— 本次要哪一个 kernel，取值是 `kernel_map` 里的名字。

```python
slot = "gemv_kernel" if m == 1 else "gemm_kernel"
```

自带实现按这个名字找到 Kernel 类，外部后端按它找到注册在同名下的 `build_kernel`。算子分几种情形，`kernel_map` 就有几个名字。

**`inputs`** —— 即将传给 kernel 的那些张量，顺序照 `signature.inputs`，一个输入占一个位置。

```python
self.get_or_build_kernel(slot, (a, b), ...)                # GEMM：两个必填输入
self.get_or_build_kernel("group_norm", (x, weight, bias), ...)  # 没传的可选输入位置上是 None
```

外部路径按它查表：设备加上每个位置的 `(dtype, shape)`。设备也算在内，因为为一块卡编译的产物可能持有那块卡上的资源。后端的 `build_kernel` 收到的也是它，每个张量转成只有 device、dtype、shape 的 `TensorSpec`，不含数据。

没传的可选输入要留下位置、值为 `None`：后端由这个值判断输入传没传，而不是数位置个数 —— 挤掉空位，只给下界的 clamp 与只给上界的就成了同一个描述。

`inputs` 漏掉当场不报错，装上后端才抛 `OpNotAvailableError` —— 这个算子于是只能用自带 kernel，外部 target 接管不了（见[安装之后：两种状态](backends.md#three-states)）。

**`key`** —— 自带 kernel 特化在什么上，只有自带这条路用（换成外部后端服务这个算子时这两个参数怎么走，见[算子层这一侧的调用](backends.md#from-op-layer)）。

```python
key=(m, n, k, a.dtype)                             # GEMM：三个维度加 dtype
key=(self._cache_key(*input_shapes), x.dtype)      # 通用写法
```

`_cache_key` 的默认实现取所有输入形状中非静态轴的尺寸，正确但可能过细 —— 一个形状编译一次。kernel 实际只依赖其中几个量时覆写它，把形状投影过去，例如 kernel 把输入当二维处理，就把前面几维乘成一个数。

**`build`** —— 怎么构造这个自带 kernel，同样只有自带这条路用。

```python
build=lambda: self.kernel_map[slot](m, n, k, a.dtype, tune=self.tune)
```

每个 `key` 只调用一次，所以编译放在这里是安全的。算子完全没有自带实现、只指望外部后端服务时，`build` 可以不传 —— 那样在没有 target 认领设备时，调用会抛 `OpNotAvailableError`。返回值可以是一个 Kernel、一组一起构造出来的 Kernel，或一个带着它们的 dataclass —— 后两种适合一次调用要 launch 多个 kernel 的算子。

### 收尾：编译边界与注册

两件事收尾，都是几行的事：

- **要支持 `torch.compile`**，得多声明一条编译边界：`forward` 只调用那个不透明算子，校验、取 kernel、launch kernel 挪进 `_eager_forward`。上面这个算子没有声明，所以 `forward` 里就是全部工作。做法见[接入 torch.compile](torch-compile.md)。
- **把算子名加进两处的导入与 `__all__`**：算子所属家族的 [`src/tileops/ops/<family>/__init__.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/ops)（类的实现位置），以及 [`src/tileops/<family>.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops)（公开路径）。缺了后者，`from tileops.<family> import ...` 拿不到这个算子，API 参考也收不到它。

## 第三步：写 kernel

kernel 类继承 [`Kernel`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/kernels/kernel_base.py)，放在 [`src/tileops/kernels/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/kernels) 下，用 TileLang 写，构造时编译、`__call__` 时启动。构造参数与调用参数照第二步 `build` 里那次构造、以及 `kernel(a, b)` 那次调用来定。

它是这六处里唯一不受 spec 约束的一处：kernel 不读 spec，spec 校验器也不检查它的内容，只登记它的路径与类名。

构造参数与调用参数的划分有一条硬性要求：**只有会被编译进生成代码的值才进构造函数。** `GemmKernel` 是这样分的：

```python
class GemmKernel(Kernel):
    def __init__(self, m, n, k, dtype, config=None, tune=False, trans_a=False, trans_b=False):
        self.kernel = _gemm_kernel(m, n, k, trans_a, trans_b, self.dtype_str)  # 这一行就编译了
        self.init_config(config, tune)      # block_m / block_n / block_k / num_stages

    def __call__(self, a, b):               # 每次调用只传张量
        ...
```

`m`、`n`、`k`、dtype 与两个布局标志进了构造函数，因为生成的代码里这些值是常量：循环边界、TMA 描述符、WGMMA 的形状都按它们展开。tile 尺寸（`block_m` 等）同理。张量本身留给 `__call__`，每次调用只换指针。

分错的代价是重新编译。decode 一步一步往前走，`seq_len` 每步 +1，batch 随 running set 变化：

```python
# 错：seq_len 进了构造函数 —— 每一步都是一个新 kernel
kernel = AttnKernel(batch, seq_len, num_heads, dtype)

# 对：只有编译期常量进构造函数，变化的量随调用传入
kernel = AttnKernel(num_heads, head_dim, dtype)
out = kernel(q, k, v)                       # seq_len 从张量形状里读
```

上一种写法下，`get_or_build_kernel` 的 `key` 里带着 `seq_len`，每步都未命中、每步都编译一次，decode 直接跑不动。

## 第四步：写测试

测试放在 [`tests/ops/`](https://github.com/tile-ai/TileOPs/tree/main/tests/ops)，比对对象就是 spec 的 `ref_api`，逐点比。形状与 dtype 取 spec 声明的范围，小形状标 `smoke` 进 PR 检查，大形状标 `full` 留给 nightly。

骨架用 [`tests/test_base.py`](https://github.com/tile-ai/TileOPs/blob/main/tests/test_base.py) 里的 `TestBase` 与 `FixtureBase`，用例写在 `PARAMS` 里。

如果这个算子有可选输入，传与不传各至少要有一条用例 —— 两侧走的往往是不同的 kernel。

## 第五步：写 benchmark

benchmark 放在 [`benchmarks/ops/`](https://github.com/tile-ai/TileOPs/tree/main/benchmarks/ops)，继承 `ManifestBenchmark`。形状不自己写，而是经 `load_workloads(<算子名>)` 从 spec 的 `workloads` 取 —— 手写形状过不了 L4 校验：

```python
from benchmarks.benchmark_base import ManifestBenchmark, workload_params
from tileops.manifest import load_workloads

_OP_NAME = "GemmFwdOp"
```

另外至少要记一个非 TileOPs 的基线，否则这一行没有比较对象。基线若需要转换输入，转换的代码留在它自己的计时区间内，不要挪出去。报出来的数字各是什么意思，见[benchmark 怎么计时](timing.md)。

## 第六步：反转实现状态，让算子进入 CI 校验

上面五处都写完之后，先跑下面三条命令自查一遍：

```bash
python scripts/validate_manifest.py --check-op GemmFwdOp   # spec 与实现一致，五级全跑
python -m pytest tests/ops/test_gemm.py -v                # 数值与 ref_api 一致
python -m pytest benchmarks/ops/bench_gemm.py             # benchmark 能出数
```

三样都过，再把 spec 的 `status` 从 `spec-only` 反转成 `implemented`。这一改动的效果是让校验从 L0 扩到五级全跑，算子由此进入 CI 的保护范围：往后每次改动，spec 校验器、测试与 nightly benchmark 都会对照 spec 检查一遍。

## 接下来

算子跑起来之后，还有两件可选的事：

- 让算子能进使用者的编译图 —— [接入 torch.compile](torch-compile.md)。
- 让它在别的硬件上由别人的 kernel 服务 —— [接入新硬件后端](backends.md)。
