# 接入 torch.compile

一个 TileOPs op 接入 `torch.compile` 之后，在使用者的编译图中成为一个节点，这个节点的形态不随服务它的 backend 变化。

接入只需要一项工作：在 op 层声明一条编译边界。边界之外由 dynamo 追踪，边界之内对编译器不可见。

正文说明接入相关的操作：

1. 判断一个 op 是否已接入；
1. 编译一段调用它的代码；
1. 调用时的五条约定；
1. 为尚未接入的 op 声明这条边界需要编写的代码。

附录说明这条边界为什么只能这样划分：dynamo 的工作方式、它与 op 层的不一致之处、边界为什么位于 op 层，以及这条边界的代价与限制。

## 调用一个已接入的 op

### 判断一个 op 是否已接入 {#supported}

读取类属性 `compile_op_names`。它非空时，说明这个类声明了编译边界（`compile_boundary = True`），边界位于 op 层，`fullgraph=True` 可用；它是空 tuple 时，说明这个类没有声明编译边界。

```python
>>> from tileops.norm import RMSNormFwdOp
>>> RMSNormFwdOp.compile_op_names
('tileops::norm_rms_norm_fwd',)
```

尚未迁移的 op 在 `fullgraph=True` 下报错，在默认设置下切图。

### 编译一段调用它的代码

构造 op 实例，再把调用它的函数传给 `torch.compile`，不需要其他步骤：

```python
import torch
from tileops.norm import RMSNormFwdOp

op = RMSNormFwdOp(normalized_shape=(4096,))     # 构造一次，反复使用

@torch.compile(fullgraph=True)
def block(x, weight):
    return op(x, weight)

x = torch.randn(2048, 4096, device="cuda", dtype=torch.float16)
w = torch.randn(4096, device="cuda", dtype=torch.float16)
block(x, w)
```

用 `TORCH_LOGS=graph_code` 运行时会打印捕获到的图，图中只有 `tileops::norm_rms_norm_fwd` 一个节点，kernel 内部的多次调用不出现在图中。

### 调用时的五条约定

五条约定各对应边界上的一处机制。违反任何一条，编译路径的行为都会与 eager 路径不同。

- **op 实例构造一次并反复使用。** 实例键是编译期常量，一个实例对应一张编译图；在循环中新建实例时，每次迭代都要重新编译。
- **stride 不会原样传递。** op 不写入的非连续输入在节点内部被转为连续张量，op 自己分配的输出总是连续张量；后续计算需要其他布局时，在 op 之外自行转换。输出就是被写入的输入（`alias`）或调用方提供的 `out` 时，沿用那个张量的存储。
- **meta 张量不能用于预热。** 声明边界之后，传入 meta 或 fake 张量的调用在 fake 函数处返回，不会执行到构造 kernel 的步骤。
- **CUDA graph 捕获之前需要预热。** 用真实张量以相同形状至少调用一次：构造 kernel 时允许编译，捕获期间只允许在缓存命中后直接调用。各阶段分别允许执行哪些操作，见[各阶段允许做什么](backends.md#phase-limits)。
- **换到另一块卡时可能重新构造 kernel。** 对由 target 服务的调用，设备是 kernel 缓存键的一部分，同一个实例换到另一块卡上会重新构造一次。in-tree kernel 的缓存键是选中实现的 `entry_for` 返回的 build identity，只有构建结果与设备有关时才包含设备。构造函数中指定的 `target=` 在首次编译调用中同样生效；构造失败不会把 op 固定到任何 target。

### 接入之后成立的三项保证

边界位于 op 层之后，调用方可以依赖以下三点：

- **编译图不随 target 变化。** 更换 backend 或硬件后，同一段代码编译出的图完全相同，编译产物因此与 backend 无关。
- **`fullgraph=True` 可用。** 前提是该 op 已经声明这条契约，判断方法见[判断一个 op 是否已接入](#supported)。
- **输出的形状、dtype 与 stride 由 manifest 规定。** 它们与 kernel 内部的分块和 padding 方式无关。op 自己分配的输出总是连续张量。

## 为新 op 声明编译边界：`RMSNormFwdOp`

本节给出接入一个 op 需要编写的代码：边界如何声明、fake 如何编写，以及 target 判定为什么要在节点内部重新执行一次。其中涉及的追踪、切图与 guard 见 [dynamo 是怎么工作的](#dynamo)。

`RMSNormFwdOp` 是仓库中第一个接入的 op。下面是它的骨架，略去 docstring，完整代码见 [`src/tileops/ops/norm/rms_norm.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/ops/norm/rms_norm.py)：

```python
class RMSNormFwdOp(Op):
    # the operators, their fakes and compile_op_names are generated from the manifest entry
    compile_boundary: ClassVar[bool] = True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {"rms_norm": RMSNormKernel}
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"rms_norm": RMSNormFwdInterface}

    def forward(self, x, weight=None):
        # the only line: call the generated operator
        return self._call_boundary(x, weight)

    def _eager_forward(self, x, weight=None):
        weight = None if weight is None else weight.contiguous()
        x = x.contiguous()                         # the generated checks have run
        call = LayerNormCall(
            device=x.device,
            n=math.prod(self.normalized_shape),
            eps=torch.finfo(torch.float32).eps if self.eps is None else float(self.eps),
            dtype=x.dtype,
        )
        return self.kernel_for("rms_norm", call)(x, weight)
```

声明只有以上内容。operator 与它的 fake 函数都从 manifest 条目生成，每个副作用分支对应一个 operator：

- 张量参数按 `signature.inputs` 的顺序排列；
- 返回值由 `signature.outputs` 决定；
- 被写入的参数恰好是标记了 `mutated` 的输入；
- 每个输出的形状与 dtype 取自签名。

operator 的名字是 `tileops::<family>_<snake(class)>`（类名本身以 family 开头时只写一次），这里就是 `tileops::norm_rms_norm_fwd`。一个分支的 operator 若写入某个输入、填写 `buffer: out` 或不产出某个输出，名字后面按这个顺序分别加上 `_writes_<输入>`、`_out`、`_without_<输出>`。op 不自行命名 operator，因此 `compile_op_names` 不可能与注册的名字不一致。

一次调用经过的各层，以及边界所在的位置：

<figure class="callpath" markdown="0">
  <div class="cp-step cp-traced"><code>Op.__call__</code><span>调用 <code>forward</code>，不判定 target</span></div>
  <div class="cp-step cp-traced"><code>forward</code><span>一行，调用不透明 operator</span></div>
  <div class="cp-boundary"><span>编译边界</span></div>
  <div class="cp-step cp-opaque"><code>生成的 operator</code><span>取回 op 实例，执行生成的检查，判定 target，失败时撤销</span></div>
  <div class="cp-step cp-opaque"><code>_eager_forward</code><span>转为连续张量、取得 kernel、launch kernel</span></div>
  <figcaption>紫色的两层在 dynamo 的追踪范围内，<code>forward</code> 中的那一行是 dynamo 追踪到的最后一处；边界以下由不透明 operator 执行，编译器看不到。</figcaption>
</figure>

其中有三处写法是固定的。

**第一处，op 实例通过字符串键取回，不直接传递对象。** schema 的类型只有 `Tensor`、`int`、`float`、`bool`、`str` 等固定几种，没有「任意 Python 对象」；而 op 体需要的 `kernel_map`、已确定的 target 与 kernel 缓存表都保存在实例上，无法拆成 schema 参数。键还有两个不能改变的细节：

- **键是字符串，不是整数。** 字符串在追踪期是常量，整数会被泛化为 `SymInt`。
- **键从不重用。** 由于键是常量，inductor 会把 fake 函数给出的形状固定在编译产物中；重用键的 op 会继承前一个实例的形状。

**第二处，fake 函数用 `torch.empty` 按签名检查推出的形状与 dtype 构造输出，不使用 `torch.empty_like(x)`。** fake 函数返回的张量在形状、dtype 与 stride 三项上都必须与真实执行的返回值一致。不一致时，要么在追踪期报错，要么在运行期按错误布局访问内存而静默出错。op 体先把输入转为连续张量，再写入新分配的输出，因此真实输出总是连续的；而 `empty_like` 会复制入参的 stride，输入不连续时，fake 函数声明的布局就是真实执行不会产出的布局。

**第三处，target 在节点内部判定，不在 `Op.__call__` 中判定。** 追踪期执行 `self.x = ...` 时，dynamo 把这次写入记为待执行的副作用，等整张图执行完才补写；而不透明节点的执行早于补写，所以节点之外刚写下的判定结果在节点之内读不到。因此以下两件事都在节点内部完成：

- 判定若写在节点之外，第一次编译调用会静默使用错误的实现。
- 判定失败时的撤销由做出判定的位置负责，因为编译产物不保留调用点的 `try/except`。

三处写法的原因相同：torch 的编译与声明机制以函数为单位，而需要编译的是一个对象上的一次调用。

## 附录：这条边界为什么是这样

### dynamo 是怎么工作的 {#dynamo}

本节说明 dynamo 如何决定哪些代码能进入编译图。一个 op 接入 `torch.compile` 需要满足的条件由此而来。

dynamo 是 `torch.compile` 的前端，工作在 CPython 的帧求值层（PEP 523）。

**dynamo 只有一个触发入口：`torch.compile`。** `torch.compile(fn)` 返回一个包装对象，包装对象被调用时才发生追踪；`nn.Module.compile()` 与装饰器写法是同一入口的另外两种形式。不经过这个入口的调用都执行原来的 Python 路径，与 dynamo 无关。下文把这种路径称为 eager。

第一次调用时，dynamo 接管这一帧，逐条符号执行字节码，把其中的张量运算记录为一张 FX 图，把无法进入图的部分留在 Python 中执行。同时，dynamo 为这张图记录一组 guard，即本次追踪所依赖的前提，例如某个张量的 dtype 与维数。此后的调用如果所有 guard 都成立，就直接重用编译产物；只要有一条不成立，就为新的情况重新追踪一次。

本页使用的三个术语含义如下：

| 术语 | 含义 |
| --- | --- |
| 编译图 | dynamo 捕获的 FX 图，一次追踪产出一张 |
| 节点 | 图中的一次 op 调用，带有输入边以及输出的形状与 dtype |
| 追踪 | 处在 dynamo 的符号执行范围内。追踪期不执行真实计算，只做记录 |

图随后交给编译 backend（inductor 等），由它完成融合、内存规划与代码生成。图越大，可融合的相邻 op 越多，因此算子库中的每个 op 都需要能作为节点出现在使用者的图中。

dynamo 有两条规则决定了接入的方式：

- **默认全部内联。** 被调用的函数本身不构成边界，函数体会并入同一次追踪。要让某一段 Python 代码不被追踪，只能显式声明。
- **无法追踪的代码有两种处理方式。** 默认设置下切图（graph break），这一段退回 Python 执行，一张图被切成多张；`fullgraph=True` 下直接报错。后者让问题在开发期暴露，因此算子库以 `fullgraph=True` 作为验收条件。

### op 层与 dynamo 的不一致之处

把上述规则应用到 TileOPs 的 op 上，接入的障碍是：dynamo 编译的单位是帧，即函数；而 TileOPs 的 op 是对象，一次调用要完成四项工作，其中只有最后一项属于图：

| 一次调用完成的工作 | 是否应当被 dynamo 捕获 |
| --- | --- |
| 校验 dtype 与形状，将输入归一为连续张量 | 不应当 |
| 判定本次调用由哪个 target 服务 | 不应当 |
| 取出或构造 kernel | 不应当，捕获到这里会失败 |
| launch kernel 并得到输出 | 应当，成为编译图中的一个节点 |

这张表需要补充三点。

**「不应当被捕获」的工作照常执行。** 四项工作在每次调用中都会执行，区别只在于是否进入编译图。

**这个区分需要人工标注**，dynamo 自己无法区分。torch 为此提供两个接口：`torch.library.custom_op` 把这一次调用注册为一个 operator，dynamo 在图中只放一个节点，不追踪其实现；`register_fake` 告诉编译器这个节点的输出是什么，它只接收输入的元信息，不接触真实数据。

**不标注时，这些代码会被追踪，并且一定失败。** 以未声明边界的 `RMSNormFwdOp` 为例，实例的两种状态都无法编译：

- 尚未构造过 kernel 的实例会在本次调用中构造 kernel，dynamo 因此追踪进构造函数中的 TileLang JIT。
- 已经构造过 kernel 的实例跳过构造，但每次调用仍要重新解析 TileLang program，dynamo 追踪进 `@tilelang.jit`，停在 `inspect.signature`。

### 为什么边界位于 op 层 {#at-op-layer}

边界可以划在 op 层，也可以划在更低的 kernel 层。两者的差别体现在使用者的编译图上。

图中那个节点的身份，包括名字、参数、粒度以及 fake 函数给出的输出，就是使用者看到的 op。边界划在 kernel 层时，更换 backend 就更换了这个节点，同一个 op 在不同 target 下会编译出不同的图，编译产物因此与 backend 绑定。边界划在 op 层时，节点的身份由 op 决定，与服务它的 backend 无关。

这个位置同时决定了 fake 函数的写法。op 层不知道外部 kernel 内部如何分块、如何 padding，对所有 target 都成立的形状规则只有 manifest 中的规则，因此 fake 函数只能依照 manifest 推导。

节点内部对编译器不可见，但它对外的契约是完整的：

- schema 给出名字与参数类型；
- fake 函数给出输出的形状、dtype、设备与 stride；
- 别名标注恰好列出它写入的入参，`RMSNormFwdOp` 没有这样的入参。

由于契约完整，边界带来的得失可以明确区分：

- **节点之间的优化照常进行。** 包括排布 buffer、计算生命周期、与无依赖的相邻节点交换顺序，以及在输出无人使用时整体删除节点。
- **节点内部的优化不再进行。** 相邻 op 无法融合进节点，输出必须写入显存。

对算子库而言，这个取舍是合理的：节点内部是 TileLang 已经编译好的 kernel，本来就不需要 inductor 介入。

### 边界的代价

以下数据在空闲的 H200 上实测，形状为 2048×4096，dtype 为 fp16；每次调用的数字取 2000 次迭代 × 9 轮、三次运行中的最小值：

| | 边界位于 kernel 层 | 边界位于 op 层 |
| --- | --- | --- |
| kernel 时间 | 0.0119 ms | 0.0117 ms |
| eager 路径每次调用 | 42.5–45.2 µs | 38.2–42.0 µs |

kernel 本身不受影响，边界位于哪一层与 kernel 如何计算无关。eager 路径快 3–5 µs，原因是边界上移之后，一次调用只需穿过一层 op 边界。

编译图一侧的代价见[为什么边界位于 op 层](#at-op-layer)：融合不跨越节点边界，节点的输出必定写入显存。

### 编译边界不提供的能力

| 不提供 | 原因 |
| --- | --- |
| 跨节点边界的融合 | 节点内部对编译器不可见，两侧的 elementwise 计算只能留在节点之外 |
| 穿过节点的 autograd | 这条调用链服务推理，fwd 与 bwd 各自是独立的 op |
| 在同一份编译产物中切换 target | target 属于 op 实例，更换 target 即更换实例，也就更换了编译图 |
| 用 meta 张量构造 kernel | 传入 meta 张量的调用只返回形状与 dtype |
