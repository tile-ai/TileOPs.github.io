# 接入新的硬件 backend

TileLang 是支持多种 backend 的 DSL，每种硬件各有一套独立的 kernel，由各自的 Python 包发行。TileOPs 因此定义了一套协议：仓库之外的 Python 包可以接管某个 op 的 kernel，取代 in-tree 实现，且不必修改 TileOPs 的任何代码。

本页说明如何接入一类新硬件，使这类设备上的 op 由该硬件自己的 kernel 执行。

**backend 只提供一件事：一个能执行这次调用的可调用对象。** 其余工作都由 op 层负责。

本页讲的是 target，即接管整个 op 的那种接入方式。前半部分按实现顺序说明 backend 作者要做的事：

1. 要写的四样东西；
2. 协议中的四个函数；
3. 一次调用如何走到这些函数；
4. 一个可直接安装运行的 backend；
5. 如何把模板改造为面向真实硬件的 backend；
6. 编写 kernel 的四条规则；
7. 各阶段允许做什么；
8. 安装之后每个 op 所处的状态，以及各条错误信息对应的原因。

后半部分说明协议为何如此设计，包括：

- 两层选择；
- op 层的契约；
- kernel 的重建条件；
- 调用方可用的接口；
- 刻意不支持的情形。

## 两种接入方式 {#two-ways}

一个仓库之外的包按自己接管的范围，从两种方式中选一种：

1. `register_kernel_type`：为一个 kernel 接口新增一个实现，与 in-tree 实现一起参与选择；
2. target：接管 op 的全部调用。

第一种方式写出的 kernel 类遵守 kernel 接口，与 in-tree 实现遵守同一份契约，写法见 [backend 如何接入](user-guide/dispatch/backends.md)。target 遵守的是 op 在 manifest 中的签名，backend 为它写一个 `build_kernel`。本页以下各节只讲 target。

## 写一个 backend 要做的四件事

| # | 做什么 |
| --- | --- |
| 1 | 在 `pyproject.toml` 中声明一条 entry point，指向 backend 模块 |
| 2 | 起一个 target 名，编写 `detect`，声明这套 kernel 面向哪一类设备 |
| 3 | 选定第一个要接管的 op，按它的 manifest 签名编写 `build_kernel` |
| 4 | 在模块顶层调用 `register_detector` 与 `register_kernel_builder` |

四件事完成后，`pip install` 即生效。下面几节依次给出：

1. 这四个函数的签名；
2. 一次调用如何走到它们；
3. 一个按这四步写成、可直接安装运行的[完整 backend](#runnable)。

之后逐个 op 增加 `build_kernel`。**目标模型用到的、自己构造 kernel 的 op 必须全部覆盖**：缺少任何一个都会报错，不会改用 in-tree 实现，因为 in-tree kernel 在该 target 的设备上无法启动。只调用子 op 的复合 op 不需要 builder。

## 协议中的四个函数

`tileops.backend` 只定义对外接口，用 Python 的结构化类型（`typing.Protocol`）表达：backend 不继承基类，也不实现抽象方法，只需写出签名相符的普通函数并注册。op 层检查返回值时同样只看结构，即 `callable()`。

backend 实现 `detect` 与 `build_kernel`，再调用 `register_detector` 与 `register_kernel_builder` 登记它们。此外还有协议定义的 `TensorSpec`，以及 `pyproject.toml` 中的一条 entry point：

| # | 名字 | 谁写 | 谁调用，何时调用 |
| --- | --- | --- | --- |
| 1 | `detect` | backend 实现 | op 层为一次调用选定 target 时，对每个 target 各调用一次 |
| 2 | `build_kernel` | backend 实现 | op 层在记忆表未命中时调用 |
| 3 | `register_detector` | backend 调用 | backend 模块被 import 时执行一次 |
| 4 | `register_kernel_builder` | backend 调用 | backend 模块被 import 时执行，每个要接管的 op 调用一次 |
| —— | `TensorSpec` | 协议定义 | op 层构造，作为 `build_kernel` 的实参传入 |
| —— | entry point | backend 在 `pyproject.toml` 中声明 | TileOPs 在构造第一个 op 时枚举 |

下面按这个顺序逐个给出签名与含义，最后给出协议定义的 `TensorSpec`。

### 1. `detect`

```python
def detect(device: torch.device) -> bool: ...
```

由 backend 实现。它回答这类设备是否由自己这套 kernel 服务，只看设备，不看 dtype 与形状。设备不属于自己时返回 `False`，不抛异常。

```python
# 认领一整类设备
def detect(device: torch.device) -> bool:
    return device.type == "acme"

# 需要读环境变量或问厂商 runtime 时也在这里做
def detect(device: torch.device) -> bool:
    if device.type != "privateuseone":
        return False
    return acme_runtime.is_present(device.index)
```

### 2. `build_kernel`

```python
def build_kernel(*inputs: "TensorSpec | None", **params) -> Callable[..., KernelResult]: ...
```

由 backend 实现，每组 `(op, target)` 一个。它的签名就是该 op 的 manifest 签名：`inputs` 按声明顺序与 `signature.inputs` 的条目一一对应，`params` 按 `signature.params` 命名。声明为 optional 的输入在本次调用中没有传入时，对应的实参是 `None`。

```python
# GroupNormFwdOp 的 spec：weight、bias 是可选输入，没传时实参是 None
def build_group_norm(x, weight, bias, *, num_groups, eps):
    if weight is None:                                   # 从槽位上的值判断传没传
        return AcmeGroupNorm(num_groups, eps, x.dtype)
    return AcmeGroupNormAffine(num_groups, eps, x.dtype)
```

### 3. `register_detector`

```python
def register_detector(target: str, detect: Callable[[torch.device], bool]) -> None: ...
```

由 backend 调用，每个 target 一次，在 backend 模块被 import 时执行。它登记该 target 的设备识别函数。

### 4. `register_kernel_builder`

```python
def register_kernel_builder(op: str, target: str, build_kernel: BuildKernel) -> None: ...
```

由 backend 调用，每个要接管的 op 一次。它登记 `(op, target)` 的 kernel 构造函数；同一组重复登记会报错。

### `TensorSpec`

```python
class TensorSpec(NamedTuple):
    device: torch.device
    dtype: torch.dtype
    shape: tuple[int, ...]
```

协议定义的类型，由 op 层构造后传入 `build_kernel`。它描述一个张量的属性，不包含张量本身。

```python
# build_kernel 收到的实参长这样
TensorSpec(device=torch.device("acme:0"), dtype=torch.float16, shape=(4096, 4096))

# 能读的就这三项
def build_gemm(a: TensorSpec, b: TensorSpec, *, trans_a, trans_b):
    m, k = a.shape                    # 形状：编译期常量，用来选实现、定 tile
    if a.dtype is not torch.float16:  # dtype：不支持就在这里报错
        raise ValueError(f"acme gemm needs fp16, got {a.dtype}")
    ...
```

返回值只需满足一条结构约定：**它必须可调用**，能以 `(*tensors)` 的形式调用，返回一个张量、一个张量元组，或在纯原地写入时返回 `None`。op 层对它的检查就是 `callable()`。

**协议只传入张量的描述。** 因此协议不需要另写一条「构造时不得读取张量内容、不得保存对张量的引用」的规则，op 层也无法校验这样的规则。这条规则要防止两件事：

- **读取数据**会使构造结果依赖数据，而记忆表只按设备与形状记录。
- **保存引用**会使张量与被缓存的 kernel 存活得一样久。

`TensorSpec` 上既没有数据也没有张量，这两件事因此无从发生。

下一节说明这四个函数在一次真实调用中分别在何时被调用。

## 一次调用如何走到 `build_kernel` {#from-op-layer}

下面列出一次调用从用户代码到 backend 的 `build_kernel` 所经过的每一步：

```python
# ── 调用方 ───────────────────────────────────────────────────────────
op = GemmFwdOp()                 # 构造时不写 target=，本次由输入张量的设备决定
                                 #   写了 target="acme" 就跳过设备探测，直接用它；
                                 #   写 target=BUILTIN 则强制走 TileOPs 自带的 kernel
a = torch.randn(4096, 4096, dtype=torch.float16, device="acme:0")
b = torch.randn(4096, 4096, dtype=torch.float16, device="acme:0")
d = op(a, b)                     # 所有输入必须在同一设备上：a.device == b.device

# ── op 层：定 target ────────────────────────────────────────────────
# 每个装好的 backend在 import 时都往注册表里放了一个 detect。op 层把 a.device
# 这一个对象原样交给每个 detect，问「这块设备是不是你这套 kernel 的」：
#   acme 的 detect(device) → True      其他 backend的 → False
#   恰好一个返回 True    → target = "acme"，这个op 实例此后固定用它
#   一个都没有返回 True  → 用 TileOPs 自带的 kernel
#   两个以上返回 True    → 抛 AmbiguousTargetError，要求显式写 target=

# ── op 层：先跑由 manifest 签名生成的检查，再把整个 op交给 target ──
#   GemmFwdOp 自己的 forward 与 kernel_for 只服务in-tree 实现，这次不走
#   传给 target 的张量顺序照 signature.inputs，不写入的输入先转成连续

# ── op 层：按设备与输入签名查外部记忆表 ─────────────────────────────
#   ("acme:0", (float16, (4096, 4096)), (float16, (4096, 4096)))
#   第一项是设备，其余每项对应一个输入的 (dtype, shape)
#   这是这个op 实例的第一次调用，表还是空的 → 未命中，往下走构造
#   同样设备、同样 dtype 与形状的下一次调用就会命中，直接跳到最后一步

# ── backend：op 层调 build_gemm，张量已转成 TensorSpec ─────────────────
#   build_gemm(TensorSpec("acme:0", float16, (4096, 4096)),
#              TensorSpec("acme:0", float16, (4096, 4096)),
#              trans_a=False, trans_b=True)      # params 按 manifest 的名字传
#   → 返回一个可调用对象

# ── op 层：存进记忆表，然后 launch ─────────────────────────────────
#   kernel(a, b)                 # d = a @ b.T，由 acme 的 kernel 算出
```

backend 只需编写其中一步，即 `build_gemm`，并把它注册进来：

```python
def build_gemm(a: TensorSpec, b: TensorSpec, *, trans_a, trans_b):
    m = a.shape[1] if trans_a else a.shape[0]
    if m == 1:                                  # 情形由 backend从 TensorSpec 自行判断
        return AcmeGemv(a, b, trans_a, trans_b)
    return AcmeGemm(a, b, trans_a, trans_b)


register_kernel_builder(op="GemmFwdOp", target="acme", build_kernel=build_gemm)
```

`build_gemm` 由 op 层调用，backend 自己从不调用它。import backend 模块时只是把它登记进注册表；它真正被调用，是在一次调用由这个 target 服务、且外部记忆表未命中的时候，每个「设备 + 输入签名」调用一次。它返回的可调用对象随后由 op 层 launch，并由 op 层存进记忆表。

这条调用路径对应以下四点：

- **`kernel_for` 与各实现的 `entry_for` 只服务 in-tree 实现。** 它们决定取哪个 in-tree kernel、按什么查表以及如何构造。op 选中某个 target 之后，整个 op 由这个 target 服务，这几处都不会执行。
- **张量按位置传入，参数按名字传入。** 在 `build_kernel(*inputs, **params)` 中，位置实参是 `TensorSpec`（没有传入的可选输入是 `None`），关键字实参是 manifest 中 `params` 的名字与本次调用的确定值。
- **一组 `(op, target)` 只注册一个 builder。** in-tree 实现内部区分的几种 kernel（GEMM 的 `kernel_types` 中有三个）不会传给 backend，`build_kernel` 从 `TensorSpec` 自行判断应返回哪个 kernel。
- **backend 不必自己缓存构建结果。** 设备与输入签名相同时，op 层不会再次调用 `build_kernel`；需要更细的区分或更少的重建时，backend 在 `build_kernel` 内部另加一层缓存。专为外部 backend 编写的 op 既不声明 `kernel_types` 也不声明 `interfaces`，没有 target 认领设备时，调用直接抛出 `OpNotAvailableError`。

## 实现一个可运行的 backend {#runnable}

[`tileops-backend-example`](https://github.com/lcy-seso/tileops-backend-example) 是按这四步写成的完整 backend，可以直接复制使用。它用纯 PyTorch 实现 kernel 并认领 CPU，因此在任何机器上都能安装、运行和测试。除了 kernel 本身不涉及专用硬件，它的其余部分与面向专用硬件的 backend 完全一致，包括 entry point、注册方式、`build_kernel` 签名、记忆规则与错误信息。

安装这个包前后的差别如下：

```console
$ python -c "import torch; from tileops.norm import RMSNormFwdOp; \
             RMSNormFwdOp(normalized_shape=(64,))(torch.randn(4,64,dtype=torch.float16), \
                                                  torch.randn(64,dtype=torch.float16))"
OpNotAvailableError: RMSNormFwdOp's in-tree kernels do not run on cpu; known targets for this op: []

$ pip install -e .

$ python -c "...同一段代码..."
# 正常返回，结果与 torch.nn.functional.rms_norm 逐位相同
```

下面按这四步逐段说明它的内容。

**第一步，`pyproject.toml` 中的三行。** entry point 组的名字固定为 `tileops.backends`，值是 backend 模块名：

```toml
[project.entry-points."tileops.backends"]
torch_cpu = "tileops_cpu"
```

`pip install` 之后不需要任何初始化。TileOPs 在构造第一个 op 时枚举这个组并 import 其中声明的模块，模块顶层的注册调用随之填好注册表。backend 既不需要继承基类，也不需要实现接口。

**第二步，起 target 名，编写 `detect`。** 两者都在 `target.py` 中。`detect` 只有一行，认领所有 CPU 设备：

```python
TARGET = "torch_cpu"


def detect(device: torch.device) -> bool:
    return device.type == "cpu"
```

两个名字含义不同：`TARGET = "torch_cpu"` 是这套 kernel 的名字，由 backend 作者决定；`device.type == "cpu"` 是它认领的设备类型，由 torch 定义。

**第三步，按 manifest 签名编写 `build_kernel`。** `RMSNormFwdOp` 的 spec 声明了两个输入 `x` 与 `weight`（可选），以及两个参数 `normalized_shape` 与 `eps`，函数的形参照抄这份声明。它与 kernel 类 `CpuRMSNorm` 一起放在 `ops/rms_norm.py` 中：

```python
def build_rms_norm(x: TensorSpec, weight: TensorSpec | None, *, normalized_shape, eps):
    if eps is None:                     # manifest 默认为 null：取参考 API 的含义
        eps = torch.finfo(torch.float32).eps
    return CpuRMSNorm(normalized_shape, eps, x.dtype)
```

**第四步，在模块顶层注册。** `ops/__init__.py` 中的 `BUILDERS` 列出这个 target 接管的全部 op，每个键都按 manifest 中的拼写书写。包的 `__init__.py` 登记一个 detector，再为表中每个 op 登记一次 builder：

```python
BUILDERS = {
    "RMSNormFwdOp": build_rms_norm,
    "GemmFwdOp": build_gemm,
}
```

```python
register_detector(target=TARGET, detect=detect)

for _op, _build_kernel in BUILDERS.items():
    register_kernel_builder(op=_op, target=TARGET, build_kernel=_build_kernel)
```

## 模板项目：结构、测试与改造

### 仓库结构

示例仓库中每个文件对应接入工作的一个环节：

| 文件 | 内容 |
| --- | --- |
| `pyproject.toml` | entry point 声明，这就是全部的安装机制 |
| `src/tileops_cpu/__init__.py` | 全部注册代码 |
| `src/tileops_cpu/target.py` | target 名与 `detect` |
| `src/tileops_cpu/ops/__init__.py` | `BUILDERS`，即这个 target 接管的全部 op。键按 manifest 中的拼写书写，键写错的 builder 永远不会被调用 |
| `src/tileops_cpu/ops/rms_norm.py`、`ops/gemm.py` | 每个 op 一个模块，存放 kernel 实现和构造它的 builder。真实 backend 在 kernel 的构造函数中编译 |
| `tests/test_takeover.py` | 数值、校验、归一与输出 |
| `tests/test_discovery.py` | entry point 与注册 |
| `tests/test_errors.py` | 三条错误路径：未登记的 op 报错而不回退，未知的 target 报错，调用失败后 op 不固定到任何 target |
| `tests/test_memoization.py` | `build_kernel` 在什么条件下被重新调用 |

`CpuRMSNorm` 在构造时**得不到行数**，这体现了「构造函数只接收编译期参数」这条规则。

### 运行测试

运行示例仓库的测试需要一个已经安装 `tileops` 的环境：

```bash
pip install -e .          # tileops 已安装时加 --no-deps
python -m pytest -q       # 两块 H200 可见与 CUDA_VISIBLE_DEVICES="" 两种情况下都是 24 passed
```

在 TileOPs 的 dev 镜像中运行同样不需要修改 TileOPs：

```bash
docker run --rm --gpus all -v "$PWD/..":/work -w /work \
  ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-afcebed1-dev \
  bash -lc 'pip install -e /work/TileOPs --no-deps -q &&
            pip install -e /work/tileops-backend-example --no-deps -q &&
            cd /work/tileops-backend-example && python -m pytest -q'
```

`tileops` 有意不写在示例的依赖列表中。这个包扩展的是一个已经存在的安装；在依赖中写上版本下限会解析到早于 `tileops.backend` 的发行版。由此产生的 `ImportError` 会被收入 `load_failures()`，表现为「这个 backend 不可用」，而真正的原因是 TileOPs 版本过旧。

### 改造成面向真实硬件的 backend

1. 复制该仓库，把 `tileops_cpu` 改为 `tileops_<硬件名>`，并相应改写 target 名。
2. 修改 `target.py` 中的 `detect`，认领对应的设备类型。
3. 把 `ops/` 下各模块中的 kernel 替换为真实 kernel：构造时编译，在 `__call__` 中启动。
4. 选定第一个要接管的 op，按它的 manifest 签名编写 `build_kernel`，并在 `BUILDERS` 中加一行。
5. [`tests/`](https://github.com/lcy-seso/tileops-backend-example/tree/main/tests) 中的四个文件大体可以直接沿用，只需替换其中的 op 名与 target 名。
6. 之后逐个 op 增加 `build_kernel`，直到覆盖目标模型用到的全部 op。

## 编写 kernel

### 签名来自 manifest

**编写 kernel 只需读 manifest，不必读 TileOPs 的源码。** builder 的签名就是该 op 的 manifest 签名。以 [`src/tileops/manifest/spec/norm.yaml`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/manifest/spec/norm.yaml) 中的 `RMSNormFwdOp` 为例：

```yaml
signature:
  forall: {B: Shape, T: "DType[float16 | bfloat16]"}
  params:                       # 按这些名字作为关键字参数传入
    normalized_shape: {type: "list[int] | tuple[int, ...]"}
    eps: {type: "float | None", default: null}
  inputs:                       # 声明顺序即传入顺序
    x: {dtype: T, shape: "[*B, *normalized_shape]"}
    weight: {dtype: T, shape: "[*normalized_shape]", optional: true}
```

对应的 builder 签名：

```python
def build_rms_norm(x: TensorSpec, weight: TensorSpec | None, *, normalized_shape, eps):
```

这个签名有两点约定：

- **参数按 op 实例保存的值传入。** 构造时没有给出 `eps` 时，builder 收到的是 manifest 的默认值 `None`，其含义与参考 API 相同，由 builder 按该语义处理；没有传入的可选输入 `weight` 收到 `None`。
- **返回值按 `signature.outputs` 的声明给出。** 单输出返回张量，多输出按声明顺序返回 tuple，纯原地写入的 op 返回 `None`。

### 构造函数只接收编译期参数

会被编译进生成代码的值（tile 尺寸、当作常量的维度、dtype）传给构造函数，其余的值留给 `__call__`。

这条规则对 decode 路径是硬性要求：`seq_len` 逐步递增，batch 随 running set 变化，把它们放进构造函数会导致每一步都重新编译。

### 形状由 manifest 规定

op 层不改变形状：kernel 收到的就是 manifest 声明的形状，kernel 需要的 layout 由它在自己的调用包装中处理。

代码与 manifest 不一致时以 manifest 为准。输出 dtype、形状规则与参数类型都由 manifest 规定，kernel 不得改写。

### kernel 无法服务这次调用时如何报错

kernel 无法服务这次调用时直接报错，不做降级处理。报错信息须给出两项内容：

- **未满足的是哪一项**：dtype、形状、arch、无可用实现，还是编译失败。
- **实际收到的值。**

只写「不支持」不构成有效的诊断信息。

## 各阶段允许做什么 {#phase-limits}

decode 路径会被 CUDA graph 捕获，因此各阶段允许执行的操作规定如下：

| 阶段 | 允许 | 不允许 |
| --- | --- | --- |
| 查记忆表（键与重建条件见 [kernel 的重建条件](#memo)） | 一次字典查找 | 其他任何操作 |
| `detect` | 一次谓词判断 | 任何 import，任何加锁 |
| 构造 kernel | 选择实现、编译、分配显存、重新 import、建立 handle | 依赖真实张量的调优 |
| 调用 kernel | 启动已编译的 kernel，经 torch allocator 分配输出 | 编译、惰性初始化、建立 handle、host 端同步 |

**模块顶层的 import 不得触发编译。** TileOPs 在构造第一个 op 时 import backend 模块，编译应当发生在 `build_kernel` 被调用时。

调用 kernel 还须满足两条与流有关的规则：

- **必须在当前流上启动。** 在 CUDA 上，当前流即 `torch.cuda.current_stream(device)`，不得改用默认流。自带 launcher 的 backend 尤其容易违反这一条。
- **内部分配的生命周期必须覆盖异步执行。** 如果只把裸指针传给 launch，对应对象必须存活到该流执行完成。协议不提供 workspace，这部分安全由 backend 自己保证。

因为构造 kernel 时允许编译，调用方需要在捕获之前完成预热，即至少执行一次同形状的非捕获调用。捕获期间只允许「查表命中后直接调用」这一条路径。

## 安装之后：三种状态 {#three-states}

`detect` 认领某一类设备之后，该设备上的**所有** op 都由这个 target 服务，缺少其中任何一个都会报错，不会改用 in-tree 实现。唯一的例外是不自己构造 kernel 的复合 op。

不回退的理由是：选中一个 target 意味着这块设备属于另一套硬件，in-tree kernel 在它上面无法启动。如果回退，只会把一条清楚的「该 target 未实现此 op」错误换成一次难以理解的启动失败。

因此安装之后，每个 op 处于以下三种状态之一：

| 状态 | 结果 |
| --- | --- |
| 该 target 为这个 op 注册了 `build_kernel` | 正常执行，整个 op 由 target 服务 |
| 没有注册，且 op 自己构造 kernel | 报错，指出这个 target 没有为该 op 注册 builder，且不会改用 in-tree 实现 |
| 没有注册，且 op 是复合 op | op 照常执行它的组合，每个子 op 各自选定 target |

因此，覆盖目标模型用到的每一个 op 是 backend 一侧的工作。op 一侧的前提由设计保证：op 层按本次调用的输入为外部路径计算记忆键（见[一次调用如何走到 `build_kernel`](#from-op-layer)）。

### 平台无关的前提

target 确定之前，op 层不查询与特定硬件绑定的信息，例如 CUDA 的 SM 版本。如果查询，在没有该驱动的机器上，调用会在到达 `build_kernel` 之前失败，而失败原因与这个 backend 无关。

如果在自己的硬件上遇到这种失败，调用栈会停在 TileOPs 内部，而不是 backend 的 `build_kernel` 中。这属于 TileOPs 一侧的回归，应提交 issue 并附上调用栈。

在看不到 GPU 的环境中，示例仓库的全部测试照常通过，没有跳过项；这次运行本身检查了这条前提。

## 错误信息与处理

以下三条均为实测输出，每条对应一种成因和一种处理方式。

**未为该 op 注册 builder：**

```
OpNotAvailableError: target 'torch_cpu' registers no kernel builder for SoftmaxFwdOp;
targets that do: []. There is no fall back to the in-tree implementation: those kernels
do not run on this target's devices.
```

处理方式是为该 op 编写并注册一个 builder。

**指定了未注册的 target：**

```
UnknownTargetError: no backend registered target 'nope'; known targets: ['torch_cpu']
```

这说明包没有安装成功，或者 target 名拼写有误。`tileops.backend.registered_targets()` 可以列出实际注册的内容。

**以 `target=BUILTIN` 强制使用 in-tree 实现：**

```
OpNotAvailableError: RMSNormFwdOp's in-tree kernels do not run on cpu; known targets for this op: ['torch_cpu']
```

`BUILTIN` 显式绕过所有 backend。in-tree 实现无法在 CPU 张量上运行，这条错误展示的正是「不改用 in-tree 实现」这条规则所要避免的后果。

**backend 包 import 失败**时，TileOPs 跳过它并发出一条警告，同时把原因收入 `load_failures()`。单个损坏的插件不会导致 TileOPs 无法导入。如果注册过程中途抛出异常，该 backend 本次注册的内容会**全部回滚**，注册表中不会留下只完成一半注册的 target。

```python
from tileops.backend import load_failures
print(load_failures())
```

## 为什么选择分两层

| 名字 | 定义 | 在 dispatch 中的位置 |
| --- | --- | --- |
| **target** | 一套 kernel 的名字。一个 backend 发行版带来一套 kernel，并为它起一个名字，例如 `"acme"` | 第一层：选中一个 target 后，本次调用的 kernel 从它这一套中选出 |
| **`detect`** | backend 编写的一个函数，每个 target 一个 | 第一层的选择依据：接收一个 `torch.device`，回答这类设备是否是自己这套 kernel 的目标设备；不是则返回 `False` |
| **`build_kernel`** | backend 为某个 op 编写的一个函数，每组 `(op, target)` 一个 | 第二层：接收本次调用的描述，即各输入张量的 device、dtype、shape 与 op 参数，从自己这套 kernel 中选定一个，构造好并返回 |

**选择分两层：TileOPs 选 target，target 在自己那套 kernel 中选一个。** 第二层发生在 `build_kernel` 内部，协议不参与，这条路径上没有 kernel 一级的概念、能力协商与候选筛选。TileOPs 实际执行的候选筛选（可用性、适用范围、优先关系）属于 in-tree 实现与范围较小的 `register_kernel_type`（见[两种接入方式](#two-ways)），target 绕过这些筛选。

`detect` 只回答设备的归属，不回答更细的问题。**本次调用是否受支持（涉及 dtype、形状与参数组合）由 `build_kernel` 回答**，因为只有它看得到完整的输入描述与参数，不支持时也在那里报错。`detect` 只拿到一个 `torch.device`，无法作出这些判断。

TileOPs 把 `torch.device` 原样传给 `detect`，自己不解析它。原因是设备类型与 target 并不一一对应，以下三种情形都会使解析出来的字符串失去意义：

- 同一个 device type 可能对应多套 kernel，分属不同厂商。
- 部分硬件经 `privateuseone` 接入，字符串中不含任何厂商信息。
- 还有一些 backend 要读取环境变量或调用厂商 runtime 才能作出判断。

## op 层的契约

以下七项是 op 层对所有 target 的契约。这些功能由 op 层实现，backend 直接重用，无需自行实现。表中各项按 backend 作者接触到它们的先后排列：

| # | op 层提供 | 对 backend 意味着什么 |
| --- | --- | --- |
| 1 | torch 侧的公开 API 与参数语义 | 这个 op 如何被调用、参数名与各参数的含义均已确定，backend 既不定义也不能改动 |
| 2 | manifest 校验 | dtype 或形状不合规的调用由 op 层拒绝，不会到达 backend |
| 3 | 参数按名字传入 | backend 收到的参数名是 manifest `params` 的名字，值是 op 实例保存的值。manifest 中默认为 null 的参数，传入的是 op 选定的值，可能是 `None`，也可能是一个具体数值 |
| 4 | 输入的连续性归一 | 本次调用不写入的输入都转成连续张量；被写入的输入按调用方传入的原样传给 backend，除非 manifest 声明它 `contiguous: true` |
| 5 | kernel 的记忆与重用 | 构造函数对每种特化调用一次：设备与输入签名相同的后续调用直接使用上一次的返回值。因此构造函数内部可以编译，op 层保证它不会被重复调用 |
| 6 | `torch.compile` 与 CUDA graph 的边界 | op 层把一次调用包装成不透明 op，并另配一个 fake，使编译器不执行也能推出输出的形状与 dtype。**backend 的 kernel 不为编译做任何事**，细节见[接入 torch.compile](torch-compile.md) |
| 7 | roofline、profile 与数值测试 | op 层已有的测试会用 backend 的 kernel 运行一遍，与 manifest 的 `ref_api` 比对数值；性能报告照常产出 |

七项均与硬件无关，每个 target 得到的完全相同。接入第三方 backend 时不得绕过其中任何一项，也不得另行实现。

TileOPs 的 in-tree kernel（[`src/tileops/kernels/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/kernels)）是**默认实现**：它没有 target 名，也不进入注册表。

**默认使用 in-tree 实现。** 没有安装 backend、没有通过 `target=` 指名、也没有设置进程默认值时，调用由 in-tree 实现服务。安装一个 backend 之后，只有当它的 `detect` 认领了这块设备，或者它被 `target=` 或 `set_default_target` 指名时，该 op 的 kernel 才换成这个 backend 的那一套。

## kernel 的重建条件 {#memo}

TileOPs 按**设备加输入签名**记住 `build_kernel` 的返回值。这个键的构成是：

> 本次调用张量所在的设备，加上按 `signature.inputs` 顺序逐项取出的 `(dtype, shape)`；声明为 optional 而本次没有传入的输入，这一项记 `None`。

因此，**设备与输入签名都相同的两次调用，TileOPs 使用同一个 kernel**，不再调用 `build_kernel`。同一个 target 的第二块卡会重新构造一次，因为为一块卡编译出的产物不一定能在另一块卡上启动。参数不进入这个键，因为它们对一个 op 实例而言是固定的。

op 层如何查表、未命中时如何调用 `build_kernel`，见[一次调用如何走到 `build_kernel`](#from-op-layer)。

由此得出两点：

- **条目不一定一直保留。** 一次调用失败时，op 会撤销它的 target 判定并清空记忆表。backend 不得假设自己返回的可调用对象一直存活，它所依赖的资源应由它自己持有引用。
- **需要更细或更粗的粒度时，都在 backend 一侧解决。** 更细的区分在 backend 内部处理；需要减少重建次数时，在 `build_kernel` 内部另加一层缓存。

## 调用方可用的接口

以下接口面向使用者，backend 作者不需要调用，但调试时会用到：

```python
from tileops.backend import (
    BUILTIN, registered_targets, set_default_target, default_target, load_failures,
)

registered_targets()                 # ['torch_cpu']
registered_targets("RMSNormFwdOp")   # ['torch_cpu']
set_default_target("torch_cpu")      # 进程默认，优先于设备探测
set_default_target(BUILTIN)          # 全局关闭替换
```

target 按以下顺序选取：

1. 构造参数 `target=`；
2. 进程默认值；
3. 设备探测。

`BUILTIN` 强制使用 in-tree 实现。指定的 target 没有注册或没有实现该 op 时直接报错，不会改用其他 target。

## 协议不支持的情形

以下情形不在这套协议的范围内，理由见表：

| 不支持 | 理由 |
| --- | --- |
| 同一个 target 上存在多个 backend | 一个 target 对应一套 kernel 与一个提供者。重复注册同一组 `(op, target)` 会直接报错，因为这说明安装了两个都声明服务该 target 的包 |
| 跨 target 回退 | 指定的 target 没有实现时直接报错，不会改用其他 target 执行 |
| backend 改变输入形状，或代替调用方还原输出 | 这是 op 层对所有 target 统一提供的服务；如需改动，就对所有 target 一起改动 |
| 一次调用跨越多个设备 | 所有输入位于同一设备上，manifest 声明 `device: cpu` 的张量除外 |
| 调用方提供 workspace 或显式 stream | backend 需要的只是当前流，而 torch 的流本身就是隐式的当前值 |
| 与 autograd 联动 | 这条调用链服务推理，fwd 与 bwd 各自是独立的 op |
