# 接入新硬件后端

本页说明如何把一类新硬件接入 TileOPs，使这类设备上的算子由自己的 kernel 执行。

TileLang 是多后端 DSL，每种硬件各有一套独立的 kernel，由各自的 Python 包发行。TileOPs 因此规定了一套协议：一个仓外的 Python 包可以接管某个算子的 kernel，取代 TileOPs 自带的实现，而接入过程不需要修改 TileOPs 的任何代码。

下文先说明算子层与后端如何分工、协议由哪几个概念构成，再介绍以 CPU 作为后端的模板项目，最后是安装之后的状态、错误信息对照与各阶段的限制。

## 算子层与后端的分工

**后端只提供一件事：一个能算这次调用的可调用对象。** 以下七项由算子层提供，对每个 target 都相同，按后端作者接触到它们的先后排列：

| # | 算子层提供 | 对后端意味着什么 |
| --- | --- | --- |
| 1 | torch 侧的公开 API 与参数语义 | 这个算子如何被调用、参数名与各参数的含义均已确定，后端既不定义也不能改动 |
| 2 | manifest 校验 | dtype 或形状不合规的调用被算子层拒绝，不会到达后端 |
| 3 | 参数规范化 | 后端收到的参数都是确定值。manifest 里写 `eps: float \| None` 的，传下来的是算子层算好的那个数，不是 `None` |
| 4 | 输入的连续性归一 | 后端只收到连续张量，不必处理非连续输入 |
| 5 | kernel 的记忆与重用 | 构造函数按特化调用一次：设备与输入签名相同的后续调用直接使用上一次的返回值。因此构造函数内部可以编译，算子层保证它不会被重复调用 |
| 6 | `torch.compile` 与 CUDA graph 的边界 | 算子层把一次调用包成不透明算子并另配一个 fake，使编译器在不执行的前提下也能推出输出的形状与 dtype。**后端的 kernel 不为编译做任何事**，细节见[接入 torch.compile](torch-compile.md) |
| 7 | roofline、profile 与数值测试 | 算子层已有的测试会用后端的 kernel 跑一遍，与 manifest 的 `ref_api` 比对数值；性能报告照常产出 |

这七项与硬件无关，任何 target 都直接得到，接入第三方后端不得绕过它们。

TileOPs 自带的 kernel（[`src/tileops/kernels/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/kernels)）是**默认实现**：它没有 target 名，也不进注册表。

**默认状态是不替换。** 没有后端认领某块设备时，这台设备上的调用走自带实现；装上一个后端、并且它认领了这块设备，该算子的 kernel 才换成后端的那一套。协议里因此不存在「默认 target」这个概念。

## 三个关键概念

| 名字 | 定义 | 在 dispatch 中的位置 |
| --- | --- | --- |
| **target** | 一套 kernel 的名字。一个后端发行版带来一套 kernel，为它起一个名字，例如 `"acme"` | 第一层：选中一个 target，本次调用的 kernel 就从它这一套里出 |
| **`detect`** | 后端写的一个函数，一个 target 一个 | 第一层怎么选：接收一块 `torch.device`，回答这类设备是不是自己这套 kernel 的目标设备；不是则返回 `False` |
| **`build_kernel`** | 后端为某个算子写的一个函数，一组 `(算子, target)` 一个 | 第二层：接收本次调用的描述，即各输入张量的 device、dtype、shape 与算子参数，在自己这套 kernel 中选定一个、构造好并返回 |

**选择分两层：TileOPs 选 target，target 在自己那套 kernel 里选一个。** 第二层发生在 `build_kernel` 内部，协议不参与，也不存在 kernel 一级的概念、能力协商与候选筛选。

`detect` 只回答设备的归属，粒度到此为止。**本次调用是否受支持 —— 涉及 dtype、形状与参数组合 —— 由 `build_kernel` 回答**，因为只有它看得到完整的输入描述与参数；不支持时在那里报错。这些判断交给 `detect` 是做不到的，它只拿到一块 `torch.device`。

TileOPs 不解析 `torch.device`，而是把它原样传给 `detect`。这样做是因为设备类型与 target 并不一一对应，有三种情形都会让解析出来的字符串失去意义：

- 同一个 device type 可能对应多套 kernel，分属不同厂商。
- 部分硬件经 `privateuseone` 接入，字符串里不含任何厂商信息。
- 还有一些后端要读取环境变量、或调用厂商 runtime 才能作出判断。

## 协议：两个注册函数与一个构造函数

`tileops.backend` 这个模块就是协议本身：它只定义对外的接口，不含任何实现。

接口以 Python 的结构化类型（protocol）表达，后端因此不继承任何基类，也不实现任何抽象方法 —— 写出签名相符的普通函数，再把它注册进来就够了。算子层对后端返回值的检查同样只看结构，即 `callable()`。

后端要用到的名字共四个，加上一条 entry point，各自的分工如下：

| 名字 | 谁写 | 谁调用，何时调用 |
| --- | --- | --- |
| `register_detector` | 后端调用 | 后端模块被 import 时执行一次 |
| `register_kernel_builder` | 后端调用 | 同上，每个要接管的算子调用一次 |
| `TensorSpec` | 协议定义 | 算子层构造，作为 `build_kernel` 的实参传入 |
| `build_kernel` | 后端实现 | 算子层在记忆表未命中时调用 |
| entry point | 后端在 `pyproject.toml` 中声明 | TileOPs 在构造第一个算子时枚举 |

四个名字的签名如下：

```python
def register_detector(target: str, detect: Callable[[torch.device], bool]) -> None:
    """登记该 target 的设备识别函数。detect(device) 回答这类设备是否由
    自己这套 kernel 服务；不是则返回 False，不要抛异常。dtype 与形状
    是否支持不在这里回答。"""


def register_kernel_builder(op: str, target: str, build_kernel: BuildKernel) -> None:
    """登记 (算子, target) 的 kernel 构造函数。同一组重复登记会报错。"""


class TensorSpec(NamedTuple):
    """一个张量是什么，不含张量本身。"""
    device: torch.device
    dtype: torch.dtype
    shape: tuple[int, ...]


def build_kernel(*inputs: "TensorSpec | None", **params) -> Callable[..., KernelResult]:
    """后端实现的构造函数，签名即该算子的 manifest 签名：inputs 与
    signature.inputs 的条目一一对应、按声明顺序，params 按
    signature.params 命名。声明为 optional 的输入本次没有传入时，
    对应的实参是 None。"""
```

返回值只需满足一条结构约定：**它必须可调用**，能以 `(*tensors)` 的形式调用，返回一个张量、一个张量元组，或纯原地写入时的 `None`。算子层对它的检查就是 `callable()`。

**协议传的是描述，不是张量。** 这样就不必再写一条「构造时不得读张量内容、不得保存对张量的引用」的规则，那条规则算子层根本无法校验。它要防的是两件事：

- **读取数据**会让构造结果依赖数据，而记忆表只按设备与形状记录。
- **保存引用**会让张量随着被缓存的 kernel 一起存活整个进程。

`TensorSpec` 上既没有数据也没有张量，这两件事于是无从写起。

## builder 签名与 manifest

**编写 kernel 只需阅读 manifest，不需要阅读 TileOPs 的源码。** builder 的签名就是该算子的 manifest 签名。以 [`src/tileops/manifest/normalization.yaml`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/manifest/normalization.yaml) 中的 `RMSNormFwdOp` 为例：

```yaml
signature:
  inputs:                       # 声明顺序即传入顺序
    x: {dtype: "float16 | bfloat16"}
    weight: {dtype: "same_as(x)"}
  params:                       # 按这些名字作为关键字参数传入
    normalized_shape: {type: "list[int] | tuple[int, ...]"}
    eps: {type: "float | None", default: null}
```

与之对应的 builder 签名是：

```python
def build_rms_norm(x: TensorSpec, weight: TensorSpec, *, normalized_shape, eps):
```

这份签名有两点要注意。

- **`eps` 收到的是 `1e-6`，而不是 `None`。** manifest 中的默认值写作 null，但算子层已经把它规范化为确定的数值。所有可选参数都是如此。
- **返回值按 `signature.outputs` 的声明给出** —— 单输出返回张量，多输出按声明顺序返回 tuple，纯原地写入的算子返回 `None`。

**构造函数只接收编译期参数。** 会被编译进生成代码的值，例如 tile 尺寸、当作常量处理的维度以及 dtype，放进构造函数；其余参数留给 `__call__`。这一条对 decode 路径是硬性要求：`seq_len` 逐步递增，batch 随 running set 变化，把它们放进构造函数就意味着每一步都要重新编译。

算子层在把张量传给 kernel 之前不改变形状：kernel 收到的就是 manifest 声明的形状，需要何种 layout 由它自己在调用包装里处理。

代码与 manifest 对同一件事都有描述时，以 manifest 为准。输出 dtype、形状规则与参数类型都由 manifest 规定，kernel 不得改写，能力不足时应当报错，而报错要说清两件事：

- **未满足的是哪一项** —— dtype、形状、arch、无可用实现，还是编译失败。
- **实际收到的值是什么。**

只写「不支持」不构成有效的诊断信息。

## kernel 的重建条件

TileOPs 按**设备加输入签名**记住 `build_kernel` 的返回值。这个键的构成是：

> 本次调用张量所在的设备，加上按 `signature.inputs` 顺序逐项取出的 `(dtype, shape)`；声明为 optional 而本次没有传入的输入，这一项记 `None`。

也就是说，**设备与输入签名都相同的两次调用，TileOPs 会把同一个 kernel 交回给后端**，不再调用 `build_kernel`。同一个 target 的第二块卡会重新构造一次，因为为一块卡编译出的产物不一定能在另一块卡上启动。参数不进入这个键，它们对一个算子实例而言是固定的。

两点由此而来：

- **记忆表有上限，条目可以在任何时候被淘汰。** 后端不得假设自己返回的可调用对象一直存活；它所依赖的资源应由它自己持有引用。
- **需要更细或更粗的粒度，都在后端一侧解决。** 更细的区分在后端内部处理；希望减少重建次数，可以在 `build_kernel` 内部另加一层缓存。

## 写一个后端要做的四件事

| # | 做什么 |
| --- | --- |
| 1 | 起一个 target 名，写 `detect`，声明这套 kernel 面向哪一类设备 |
| 2 | 挑第一个要接管的算子，照它的 manifest 签名写 `build_kernel` |
| 3 | 在模块顶层调用 `register_detector` 与 `register_kernel_builder` |
| 4 | 在 `pyproject.toml` 里声明一条 entry point 指向这个模块，`pip install` 后自动生效 |

之后逐个算子增加 `build_kernel`。**目标模型用到的算子必须全部覆盖** —— 缺少任何一个都会报错，不会改用 TileOPs 自带的实现，因为那些 kernel 在该 target 的设备上启动不了。

## 模板项目：以 CPU 作为后端

[`tileops-backend-example`](https://github.com/lcy-seso/tileops-backend-example) 是按上述四步写成的一个完整后端。它以纯 PyTorch 实现 kernel、认领 CPU，因此在任何机器上都能安装、运行和测试；除 kernel 本身不涉及专用硬件之外，其余各部分 —— entry point、注册方式、`build_kernel` 签名、记忆规则、错误信息 —— 与一个面向专用硬件的后端完全一致。

安装这个包前后的差别如下：

```console
$ python -c "import torch; from tileops.ops.norm.rms_norm import RMSNormFwdOp; \
             RMSNormFwdOp(normalized_shape=(64,))(torch.randn(4,64,dtype=torch.float16), \
                                                  torch.randn(64,dtype=torch.float16))"
ValueError: RMSNormKernel is a CUDA kernel; got x on cpu and weight on cpu.

$ pip install -e .

$ python -c "...同一段代码..."
# 正常返回，结果与 torch.nn.functional.rms_norm 逐位相同
```

### 最小实现

一个后端需要提供的内容只有两部分。第一部分是 `pyproject.toml` 中的三行声明：

```toml
[project.entry-points."tileops.backends"]
torch_cpu = "tileops_cpu"
```

第二部分是模块顶层的两次注册：

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

注册代码中出现的两个名字含义不同：`target="torch_cpu"` 是这一套 kernel 的名字，由后端作者决定；`device.type == "cpu"` 是它认领的设备类型，由 torch 定义。

执行 `pip install` 之后，这个后端即自动生效：TileOPs 在构造第一个算子时枚举该 entry point 组，import 其中声明的模块，模块顶层的两次调用把注册表填好。除此之外没有其他初始化步骤，既没有需要继承的基类，也没有需要实现的接口。

### 仓库结构

示例仓库中每个文件对应接入工作的一个环节：

| 文件 | 内容 |
| --- | --- |
| `pyproject.toml` | entry point 声明，也就是全部安装机制 |
| `src/tileops_cpu/__init__.py` | 全部注册代码 |
| `src/tileops_cpu/kernels.py` | kernel 实现，真实后端在此处编译 |
| `src/tileops_cpu/pending.py` | 一个已经注册、但当前调用不到的 builder，见[下文](#three-states) |
| `tests/test_takeover.py` | 数值、校验、归一与输出 |
| `tests/test_discovery.py` | entry point 与注册 |
| `tests/test_errors.py` | 四条错误路径 |
| `tests/test_memoization.py` | `build_kernel` 在什么条件下被重新调用 |

其中 `CpuRMSNorm` 在构造时**得不到行数**，正是「构造函数只接收编译期参数」这一条的体现。

### 运行测试

运行示例仓库的测试需要一个已经安装 `tileops` 的环境：

```bash
pip install -e .          # tileops 已安装时加 --no-deps
python -m pytest -q       # 有 GPU：22 passed；无 GPU：20 passed, 2 skipped
```

在 TileOPs 的 dev 镜像中运行同样不需要修改 TileOPs：

```bash
docker run --rm --gpus all -v "$PWD/..":/work -w /work \
  ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-afcebed1-dev \
  bash -lc 'pip install -e /work/TileOPs --no-deps -q &&
            pip install -e /work/tileops-backend-example --no-deps -q &&
            cd /work/tileops-backend-example && python -m pytest -q'
```

`tileops` 刻意不写在示例的依赖列表中。这个包扩展的是一个已经存在的安装，而在依赖中写上版本下限会解析到早于 `tileops.backend` 的发行版；由此产生的 `ImportError` 会被收入 `load_failures()`，呈现出来的是「这个后端不可用」，而真正的原因是 TileOPs 版本过旧。

### 改造成面向真实硬件的后端

1. 复制该仓库，把 `tileops_cpu` 改为 `tileops_<硬件名>`，target 名同样改写。
2. 修改 `_detect`，认领对应的设备类型。
3. 把 `kernels.py` 替换为真实 kernel，构造时编译，`__call__` 时启动。
4. 选定第一个要接管的算子，照它的 manifest 签名编写 `build_kernel`。
5. [`tests/`](https://github.com/lcy-seso/tileops-backend-example/tree/main/tests) 中的四个文件大体可以直接沿用，替换其中的算子名与 target 名即可。
6. 之后逐个算子增加 `build_kernel`，直到覆盖目标模型用到的全部算子。

## 安装后的三种状态 {#three-states}

一旦 `detect` 认领了某一类设备，该设备上的**所有**算子都由这个 target 服务，其中任何一个缺失都会报错，不会改用 TileOPs 自带的实现。

不回退的理由是：选中一个 target 就意味着这块设备属于另一套硬件，自带的 kernel 在它上面根本启动不了。真去回退，只会把一条清楚的「该 target 未实现此算子」换成一次难以理解的启动失败。

因此安装之后，每个算子处于以下三种状态之一：

| 状态 | 例子 | 结果 |
| --- | --- | --- |
| 已注册 builder，且该算子在取 kernel 处传入了张量 | `RMSNormFwdOp` | 正常执行 |
| 已注册 builder，但该算子尚未在取 kernel 处传入张量 | `GemmOp` | 报错，**需要修改的是 TileOPs** |
| 未注册 builder | 其余全部算子 | 报错 |

第二种状态源于 TileOPs 当前的迁移进度：算子取 kernel 的位置必须把即将传给该 kernel 的张量一并传入，TileOPs 才能计算外部路径的记忆键。

```python
# TileOPs 内部，算子自身的 forward
self.get_or_build_kernel("gemm_kernel", (a, b), key=..., build=...)
#                                       ^^^^^^ 这一项
```

截至 2026 年 8 月，只有 `RMSNormFwdOp` 传入了这一项，其余 84 处取 kernel 的位置尚未传入。这是一项机械改动，会逐个算子补齐。

示例仓库 `src/tileops_cpu/pending.py` 中的 `build_gemm` 正是这种情况：实现正确，注册成功，但目前调用不到。在 TileOPs 补上这一项之前先把 builder 写好是正常的工作顺序 —— `GemmOp` 可用的当天，这个后端无需任何改动即可服务它。

### 平台判据

即使一个算子已经能够走到外部路径，TileOPs 的 `forward` 中仍可能残留与特定硬件绑定的代码。`GemmOp` 就属于这种情况：

```
tileops/ops/gemm.py:102       _get_kernel
tileops/kernels/call_spec.py  CallSpec.__post_init__
tileops/utils/utils.py:39     get_sm_version  ->  torch.cuda.current_device()
```

选中的 target 面向 CPU，这段代码却仍然去查询 CUDA 的 SM 版本。在没有 CUDA 驱动的机器上，调用在到达 `build_kernel` 之前就已经失败。

截至 2026 年 8 月，TileOPs 中约有 94 处这类判据。清除它们是接入第一个真实异构后端的前提，将按 family 分批进行。在此之前，后端作者会在自己的硬件上遇到它们；遇到时应向 TileOPs 提 issue 并附上调用栈，需要修改的是 TileOPs。

示例仓库因此为两个测试加上了 `requires_cuda_runtime` 标记，在没有 GPU 的机器上自动跳过。这两个测试检验的是 TileOPs 的平台假设，而不是这个后端。

## 错误信息对照

以下四条均为实测输出，分别对应一种成因和一种处理方式。

**已注册 builder，但该算子尚未在取 kernel 处传入张量：**

```
OpNotAvailableError: target 'torch_cpu' serves GemmOp, but its 'gemm_kernel' call site
does not hand over the tensors a builder is described with; that op is not wired to
external targets yet
```

这是 TileOPs 一侧的缺口，不是后端的问题。可以等待它补上 `inputs=`，或者提 issue 请求优先处理该算子。

**未为该算子注册 builder：**

```
OpNotAvailableError: target 'torch_cpu' registers no kernel builder for SoftmaxFwdOp;
targets that do: []. There is no fall back to the in-tree implementation: those kernels
do not run on this target's devices.
```

为该算子编写并注册一个 builder。

**指定了未注册的 target：**

```
UnknownTargetError: no backend registered target 'nope'; known targets: ['torch_cpu']
```

说明包没有安装成功，或者 target 名拼写有误。可以用 `tileops.backend.registered_targets()` 查看实际注册的内容。

**以 `target=BUILTIN` 强制使用 TileOPs 自带的实现：**

```
ValueError: RMSNormKernel is a CUDA kernel; got x on cpu and weight on cpu.
Another target's backend serves other devices.
```

`BUILTIN` 显式绕过所有后端。自带实现无法在 CPU 张量上运行，这条错误正说明了「不改用自带实现」这条规则所要避免的后果。

**后端包 import 失败**时，TileOPs 会跳过它并发出一条警告，同时把原因收入 `load_failures()`。单个损坏的插件不会导致 TileOPs 无法导入。如果注册过程中途抛出异常，该后端本次注册的内容会**全部回滚**，注册表中不会留下一个只实现了一半的 target。

```python
from tileops.backend import load_failures
print(load_failures())
```

## 调用方 API

以下接口面向使用者，后端作者不需要调用，但调试时用得上：

```python
from tileops.backend import (
    BUILTIN, registered_targets, set_default_target, default_target, load_failures,
)

registered_targets()                 # ['torch_cpu']
registered_targets("RMSNormFwdOp")   # ['torch_cpu']
set_default_target("torch_cpu")      # 进程默认，优先于设备探测
set_default_target(BUILTIN)          # 全局关闭替换
```

target 的选取顺序是：构造参数 `target=`，其次是进程默认值，最后是设备探测。`BUILTIN` 强制走 TileOPs 自带的实现。指定的 target 没有注册或没有实现该算子即报错，不会改用其他 target。

## 各阶段的限制

decode 路径会被 CUDA graph 捕获，因此各阶段允许执行的操作分别规定如下：

| 阶段 | 允许 | 不允许 |
| --- | --- | --- |
| 查记忆表 | 一次字典查找 | 其他任何操作 |
| `detect` | 一次谓词判断 | 任何 import，任何加锁 |
| 构造 kernel | 选择实现、编译、分配显存、重新 import、建立 handle | 依赖真实张量的调优 |
| 调用 kernel | 启动已编译的 kernel，经 torch allocator 分配输出 | 编译、惰性初始化、建立 handle、host 端同步 |

**模块顶层的 import 不得触发编译。** TileOPs 在构造第一个算子时 import 后端模块，编译应当发生在 `build_kernel` 被调用的时候。

调用 kernel 还须满足两条与流有关的规则：

- **必须在当前流上启动**，在 CUDA 上即 `torch.cuda.current_stream(device)`，不得改用默认流。自带 launcher 的后端尤其容易违反这一条。
- **内部分配的生命周期必须跨越异步执行。** 如果只把裸指针传给 launch，对象必须存活到该流执行完成为止。协议不提供 workspace，这一部分安全由后端自己保证。

调用方需要在捕获之前完成预热，即至少执行一次同形状的非捕获调用，因为构造 kernel 允许编译。捕获期间只允许「查表命中后直接调用」这一条路径。

## 协议不支持的情形

以下情形不在这套协议的范围内，各有其理由：

| 不支持 | 理由 |
| --- | --- |
| 同一个 target 上存在多个后端 | 一个 target 对应一套 kernel、一个提供者。重复注册同一组 `(算子, target)` 会直接报错，因为这说明安装了两个都声明服务该 target 的包 |
| 跨 target 回退 | 指定的 target 没有实现即报错，不会改用其他 target 执行 |
| 整体替换一个组合算子 | 组合算子的计算发生在它构造的 sub-op 中，替换应当发生在那一层 |
| 后端改变输入形状，或代替调用方还原输出 | 这是算子层对所有 target 统一提供的服务；要改动就对所有 target 一起改动 |
| 一次调用跨越多个设备 | CPU 标量以参数形式传入，而非张量输入，因此所有输入必须位于同一设备 |
| 调用方提供 workspace 或显式 stream | 后端需要的只是当前流，而 torch 的流本身就是隐式的当前值 |
| 与 autograd 联动 | 这条调用链服务推理，fwd 与 bwd 各自是独立的算子 |
