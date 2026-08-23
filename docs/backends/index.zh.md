# 接入新后端

一个后端是可安装的 Python 包，它在自己认领的设备上接管 TileOPs 的算子：kernel 由它提供，op 层仍由 TileOPs 负责 —— manifest 校验、参数规范化、kernel 的记忆与重用、roofline、数值测试。接入过程不需要修改 TileOPs 的任何代码。

[`tileops-backend-example`](https://github.com/lcy-seso/tileops-backend-example) 是面向外部后端作者的参照实现：一个可安装、可运行、可测试的包，以纯 PyTorch 实现 kernel，接管 CPU 上的算子。除 kernel 本身不涉及专用硬件之外，其余各部分 —— entry point、注册、`build_kernel` 签名、记忆规则、错误信息 —— 与一个面向专用硬件的后端完全一致。

安装前后的差别：

```console
$ python -c "import torch; from tileops.ops.norm.rms_norm import RMSNormFwdOp; \
             RMSNormFwdOp(normalized_shape=(64,))(torch.randn(4,64,dtype=torch.float16), \
                                                  torch.randn(64,dtype=torch.float16))"
ValueError: RMSNormKernel is a CUDA kernel; got x on cpu and weight on cpu.

$ pip install -e .

$ python -c "...同一段代码..."
# 正常返回，结果与 torch.nn.functional.rms_norm 逐位相同
```

## 最小实现

一个后端需要提供的全部内容如下。`pyproject.toml` 中三行：

```toml
[project.entry-points."tileops.backends"]
torch_cpu = "tileops_cpu"
```

以及模块顶层的两次注册：

```python
from tileops.backend import (
    TensorSpec,
    register_detector,
    register_kernel_builder,
)
from .kernels import CpuRMSNorm

register_detector(
    target="torch_cpu",
    detect=lambda device: device.type == "cpu",
)


def build_rms_norm(
    x: TensorSpec,
    weight: TensorSpec,
    *,
    normalized_shape,
    eps,
):
    return CpuRMSNorm(normalized_shape, eps, x.dtype)


register_kernel_builder(
    op="RMSNormFwdOp",
    target="torch_cpu",
    build_kernel=build_rms_norm,
)
```

`pip install` 之后自动生效：TileOPs 在构造第一个 Op 时枚举这个 entry point 组，import 声明的模块，顶层的两次调用把注册表填好。没有其他初始化步骤，没有需要继承的基类，也没有需要实现的接口。

## target 名与设备类型

上面那行注册里出现了两个名字，它们含义不同：

| 名字 | 含义 | 由谁决定 |
| --- | --- | --- |
| `target="torch_cpu"` | 这套 kernel 的名字 | 后端作者 |
| `device.type == "cpu"` | 认领哪类设备 | torch |

两者不是一一对应的：同一个 device type 可能对应不止一套 kernel，分属不同厂商；部分硬件经 `privateuseone` 接入，字符串中不含任何厂商信息；有的后端还需读环境变量或调用厂商 runtime 才能确定。因此 TileOPs 不解析 `torch.device`，而是原样传给 `detect`，由后端回答。

`detect` 只回答设备归属。**某次调用是否受支持 —— dtype、形状、参数组合 —— 由 `build_kernel` 回答**，只有它能看到这些信息。对不认领的设备返回 `False`，不要抛异常。

## 示例仓库结构

| 文件 | 内容 |
| --- | --- |
| `pyproject.toml` | entry point 声明，即全部安装机制 |
| `src/tileops_cpu/__init__.py` | 全部注册代码 |
| `src/tileops_cpu/kernels.py` | kernel 实现。真实后端在此编译 |
| `src/tileops_cpu/pending.py` | 一个已注册但当前无法被调用的 builder，见[下文](#安装后的三种状态) |
| `tests/test_takeover.py` | 数值、校验、归一、输出 |
| `tests/test_discovery.py` | entry point 与注册 |
| `tests/test_errors.py` | 四条错误路径 |
| `tests/test_memoization.py` | 何时重新调用 `build_kernel` |

## builder 签名

一个 builder 收到各输入的描述，返回一个可调用体。它的签名由 op 的 manifest 规定，返回值在调用时收到真实张量，何时重新调用它由 TileOPs 决定。

**编写 kernel 只需阅读 manifest，不需要阅读 TileOPs 源码。**

`src/tileops/manifest/normalization.yaml` 中的 `RMSNormFwdOp`：

```yaml
signature:
  inputs:                       # 声明顺序即传入顺序
    x: {dtype: "float16 | bfloat16"}
    weight: {dtype: "same_as(x)"}
  params:                       # 按这些名字作为关键字参数传入
    normalized_shape: {type: "list[int] | tuple[int, ...]"}
    eps: {type: "float | None", default: null}
```

对应的 builder：

```python
def build_rms_norm(x: TensorSpec, weight: TensorSpec, *, normalized_shape, eps):
```

两点需要注意：

- **`eps` 收到的是 `1e-6`，不是 `None`。** manifest 中的默认值是 null，但 op 层已将其规范化为确定的数值。所有可选参数都是如此。
- **`TensorSpec` 是描述而非张量**，只有 `device` / `dtype` / `shape`，既没有数据，也没有对张量的引用。这样两类错误就无法写出来：一是根据数据内容决定构造哪个 kernel（而记忆表按形状索引，后续会取到错误的 kernel），二是让某个张量随被缓存的 kernel 存活整个进程。

对返回值只有一条要求：**可调用**。调用时按同样顺序收到真实张量，返回值按 `signature.outputs` —— 单输出返回张量，多输出按顺序返回 tuple，纯原地写返回 `None`。

**构造签名只接收编译期参数。** 会被编译进生成代码的值（tile 尺寸、当作常量的维度、dtype）放进构造函数，其余留给 `__call__`。这一条对 decode 是硬性要求：`seq_len` 逐步递增，batch 随 running set 变化，它们进入构造函数就意味着每步重新编译。`kernels.py` 中的 `CpuRMSNorm` 在构造时**拿不到行数**，原因即在于此。

## op 层的职责

以下工作对所有 target 相同，后端**不要重复实现**：

- manifest 的 dtype 校验与形状规则
- 参数规范化（可选值落实为确定值）
- 输入的连续性归一 —— 传入 kernel 的都是连续张量
- kernel 的记忆与重用
- roofline、profile、数值测试

**op 层不在把张量传给 kernel 之前改变形状。** kernel 收到的是 manifest 声明的形状；需要何种 layout 由 kernel 自行处理，在它自己的调用包装里完成。

**代码与 manifest 都描述了的事情，以 manifest 为准。** 输出 dtype、形状规则、参数类型均由 manifest 规定，kernel 不得改写；能力不足时报错。错误信息必须指明哪一项不满足（dtype / 形状 / arch / 无可用实现 / 编译失败）以及实际收到的值 —— 仅写「不支持」不构成有效诊断。

## 何时重建 kernel

TileOPs 按**输入签名**记住 `build_kernel` 的返回值：

> 按 `signature.inputs` 的顺序，逐个取 `(dtype, shape)`。

也就是说：**输入签名相同的两次调用，TileOPs 会交给后端同一个 kernel。**

`device` 不进入 key —— 产物是否随设备而不同属于后端内部的事；返回的可调用体每次都能拿到真实张量，需要时在那时分派。`params` 也不进入 key，它们对一个 op 实例是固定的。

需要更细的区分，在后端内部处理；需要更粗的粒度以减少重建，在 `build_kernel` 内部另加缓存。两个方向都在后端一侧解决。

`tests/test_memoization.py` 逐项验证了这条规则，包括 key 的实际形态：

```python
assert key == ((torch.float16, (4, 64)), (torch.float16, (64,)))  # x 在前，weight 在后
```

## 安装后的三种状态

一旦 `detect` 认领了某类设备，该设备上的**所有** op 都由这个 target 服务；缺少任何一个都会报错，**不会改用仓内实现**。原因很直接：选中一个 target 意味着该设备属于另一套硬件，TileOPs 自带的 kernel 在其上无法启动，改用仓内实现只会把一个清楚的「该 target 未实现此 op」换成一个难以理解的启动失败。

| 状态 | 例子 | 结果 |
| --- | --- | --- |
| 已注册 builder，且该 op 在取 kernel 处传入了张量 | `RMSNormFwdOp` | 正常执行 |
| 已注册 builder，但该 op 尚未在取 kernel 处传入张量 | `GemmOp` | 报错，**需要修改的是 TileOPs** |
| 未注册 builder | 其余全部 | 报错 |

第二种状态源于 TileOPs 当前的迁移进度：op 的取 kernel 处必须把要传给该 kernel 的张量一并传入，TileOPs 才能计算外部路径的记忆 key。

```python
# TileOPs 内部，op 自身的 forward
self.get_or_build_kernel("gemm_kernel", (a, b), key=..., build=...)
#                                       ^^^^^^ 这一项
```

**目前只有 `RMSNormFwdOp` 传入了它**，其余 84 个取 kernel 处尚未传入。这是机械改动，会逐个 op 补齐。

`src/tileops_cpu/pending.py` 中的 `build_gemm` 正是这种情况：实现正确，注册成功，但目前调用不到。在 TileOPs 补上这一项之前先写好 builder 是正常的工作顺序 —— `GemmOp` 可用的当天，这个后端无需改动即可服务它。

### 平台判据

即使一个 op 已经走得到外部路径，TileOPs 的 `forward` 中仍可能残留与特定硬件绑定的代码。`GemmOp` 就是这样：

```
tileops/ops/gemm.py:102       _get_kernel
tileops/kernels/call_spec.py  CallSpec.__post_init__
tileops/utils/utils.py:39     get_sm_version  ->  torch.cuda.current_device()
```

选中的 target 是 CPU 的，这段代码仍会去查询 CUDA 的 SM 版本。在没有 CUDA 驱动的机器上，调用在到达 `build_kernel` 之前就失败了。

TileOPs 中现存约 94 处此类判据，清除它们是接入第一个真实异构后端的前提，将按 family 分批进行。在此之前，后端作者会在自己的硬件上遇到这些判据。遇到时应向 TileOPs 提 issue 并附上调用栈，需要修改的是 TileOPs。

示例仓库因此为两个测试加了 `requires_cuda_runtime` 标记，在无 GPU 的机器上自动跳过。它们检验的是 TileOPs 的平台假设，而非这个后端。

## 错误信息对照

以下均为实测输出。

**已注册 builder，但该 op 尚未在取 kernel 处传入张量：**

```
OpNotAvailableError: target 'torch_cpu' serves GemmOp, but its 'gemm_kernel' call site
does not hand over the tensors a builder is described with; that op is not wired to
external targets yet
```

这是 TileOPs 一侧的缺口，不是后端的问题。等待它补上 `inputs=`，或提 issue 请求优先处理该 op。

**未为该 op 注册 builder：**

```
OpNotAvailableError: target 'torch_cpu' registers no kernel builder for SoftmaxFwdOp;
targets that do: []. There is no fall back to the in-tree implementation: those kernels
do not run on this target's devices.
```

为该 op 编写并注册一个 builder。

**指定了未注册的 target：**

```
UnknownTargetError: no backend registered target 'nope'; known targets: ['torch_cpu']
```

包未安装成功，或 target 名拼写错误。用 `tileops.backend.registered_targets()` 查看实际注册的内容。

**用 `target=BUILTIN` 强制使用仓内实现：**

```
ValueError: RMSNormKernel is a CUDA kernel; got x on cpu and weight on cpu.
Another target's backend serves other devices.
```

`BUILTIN` 显式绕过后端。CPU 张量上仓内实现无法运行，这正好说明了「不改用仓内实现」这条规则要避免的是什么。

**后端包 import 失败：** TileOPs 会跳过它并发出一条警告，把原因收进 `load_failures()`。单个损坏的插件不会导致 TileOPs 无法导入。若注册过程中途抛出异常，该后端本次注册的内容会**全部回滚**，注册表中不会留下一个只实现了一半的 target。

```python
from tileops.backend import load_failures
print(load_failures())
```

## 调用方 API

后端作者不需要调用这些，但调试时有用：

```python
from tileops.backend import (
    BUILTIN, registered_targets, set_default_target, default_target, load_failures,
)

registered_targets()                 # ['torch_cpu']
registered_targets("RMSNormFwdOp")   # ['torch_cpu']
set_default_target("torch_cpu")      # 进程默认，优先于设备探测
set_default_target(BUILTIN)          # 全局关闭替换
```

target 的选取顺序：构造参数 `target=` → 进程默认 → 设备探测。不存在「默认 target」这一概念，默认状态是不替换。

## 运行测试

需要一个已安装 `tileops` 的环境。

```bash
pip install -e .          # tileops 已安装时加 --no-deps
python -m pytest -q       # 有 GPU：22 passed；无 GPU：20 passed, 2 skipped
```

在 TileOPs 的 dev 镜像中运行，同样不需要修改 TileOPs：

```bash
docker run --rm --gpus all -v "$PWD/..":/work -w /work \
  ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-afcebed1-dev \
  bash -lc 'pip install -e /work/TileOPs --no-deps -q &&
            pip install -e /work/tileops-backend-example --no-deps -q &&
            cd /work/tileops-backend-example && python -m pytest -q'
```

`tileops` 刻意不在示例的依赖列表中：本包扩展的是一个已经存在的安装，而在依赖中写版本下限会解析到早于 `tileops.backend` 的发行版，由此产生的 `ImportError` 会被收进 `load_failures()`，读起来像是「这个后端坏了」，而非「TileOPs 版本过旧」。

## 各阶段的限制

decode 路径会被 CUDA graph 捕获，各阶段允许做的事因此分别规定：

| 阶段 | 可以 | 不可以 |
| --- | --- | --- |
| 查记忆表 | dict 查找 | 其余一切 |
| `detect` | 一次谓词判断 | 任何 import、任何加锁 |
| 构造 kernel | 选实现、编译、分配、重 import、建 handle | 依赖真实张量的调优 |
| kernel 调用 | 启动已编译的 kernel；经 torch allocator 分配输出 | 编译、惰性初始化、建 handle、host 同步 |

**模块顶层 import 不得触发编译。** TileOPs 在构造第一个 Op 时 import 后端模块，编译应发生在 `build_kernel` 被调用时。

kernel 调用还须满足两条与流相关的规则：

- **在当前流上启动**（CUDA 下即 `torch.cuda.current_stream(device)`），不得改用默认流。自带 launcher 的后端尤其容易违反这一条。
- **内部分配的生命周期必须跨越异步执行。** 只把裸指针传给 launch 时，对象必须存活到该流执行完成。协议不提供 workspace，这份安全由后端负责。

调用方须在捕获前完成预热（至少一次同形状的非捕获调用），因为构造 kernel 允许编译。捕获期间只允许走「查表命中、直接调用」这条路。

## 不支持的情形

| 不支持 | 理由 |
| --- | --- |
| 同一个 target 上有多个后端 | 一个 target 对应一套 kernel、一个提供者。重复注册同一个 `(op, target)` 直接报错，那意味着安装了两个都自称是它的包 |
| 整体替换一个组合 op | 组合 op 的计算在它构造的 sub-op 中，替换发生在那一层 |
| 后端改变输入形状，或代替调用方还原输出 | 那是 op 层对所有 target 提供的服务；要改就对所有 target 一起改 |
| 一次调用跨多个设备 | CPU 标量走 params 而非张量输入，所有输入必须在同一设备 |
| 调用方提供 workspace 或显式 stream | 后端需要的只是当前流，而 torch 的流是隐式当前值 |
| autograd 联动 | 这条链服务推理。fwd / bwd 各是独立的 op |
| 换用另一个 target | 指定的 target 没有实现就报错，不会改用别的 target 执行 |

## 作为模板使用

1. 复制该仓库，把 `tileops_cpu` 改为 `tileops_<硬件名>`，target 名同理。
2. 修改 `_detect`，认领对应的设备类型。
3. 把 `kernels.py` 替换为真实 kernel —— 构造时编译，`__call__` 时启动。
4. 选定第一个要接管的 op，照它的 manifest 签名编写 `build_kernel`。
5. `tests/` 中的四个文件基本可以直接沿用，替换 op 名与 target 名即可。
6. 之后逐个 op 增加 `build_kernel`。**目标模型用到的 op 需要全部覆盖**，缺少任何一个都会报错。
