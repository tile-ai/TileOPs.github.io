# 接入 torch.compile

把一个 TileOPs 算子接入 `torch.compile`，是让它在使用者的编译图里成为一个节点，而这个节点的形态不随服务它的后端变化。

要做的只有一件事：在算子层声明一条编译边界，界外交给 dynamo 追踪，界内对编译器不可见。

正文讲要做的事：判断一个算子是否已接入、编译一段调用它的代码、调用时的五条约定，以及给尚未接入的算子声明这条边界要写什么。

附录讲这条边界为什么只能这样划：dynamo 怎么工作、它与算子层在哪里错位、边界为什么落在算子层，以及这条边界的代价与限制。

## 调用一个已接入的算子

### 判断一个算子是否已接入 {#supported}

读类属性 `compile_op_names`：非空说明边界已经在算子层，`fullgraph=True` 可用；空 tuple 说明尚未迁移。

```python
>>> from tileops.norm import RMSNormFwdOp
>>> RMSNormFwdOp.compile_op_names
('tileops::norm_rms_norm_fwd',)
```

尚未迁移的算子在 `fullgraph=True` 下报错，默认设置下切图。

### 编译一段调用它的代码

构造算子实例，把调用它的函数交给 `torch.compile`，没有别的步骤：

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

用 `TORCH_LOGS=graph_code` 运行会打印捕获到的图：里面是 `tileops::norm_rms_norm_fwd` 一个节点，不是 kernel 内部的多次调用。

### 调用时要遵守的五条约定

五条各对应边界上的一处机制，违反任何一条，编译路径的行为就与 eager 路径不同。

- **算子实例构造一次并反复使用。** 实例键是编译期常量，一个实例对应一张编译图；在循环里新建实例，每次迭代都要重新编译。
- **不要依赖 stride 原样传递。** 非连续输入在节点内部连续化，输出恒为连续张量；后续计算需要别的布局，在算子之外自行转换。
- **不能用 meta 张量预热。** 有了边界，传入 meta 或 fake 张量的调用就在 fake 处返回，走不到构造 kernel 那一步。
- **CUDA graph 捕获之前先行预热。** 用真实张量、相同形状至少调用一次：构造 kernel 允许编译，捕获期间只允许查表命中后直接调用。各阶段分别允许执行哪些操作，见[各阶段允许做什么](backends.md#phase-limits)。
- **每个设备各自构造 kernel。** 设备是 kernel 记忆键的一部分，同一个实例换一块卡就重新构造一次。构造函数里指名的 `target=` 在首次编译调用中同样生效；构造失败不会把算子固定到任何 target。

### 接入之后成立的三项保证

边界落在算子层之后，调用方可以依赖三点。

- **编译图不随 target 变化。** 换后端或换硬件，同一段代码编出的图完全相同，编译产物因此与后端无关。
- **`fullgraph=True` 可用。** 前提是该算子已经声明这条契约，判定见[判断一个算子是否已接入](#supported)。
- **输出的形状、dtype 与 stride 由 manifest 规定。** 与 kernel 内部怎么分块、怎么 padding 无关。输入在节点内部连续化，输出恒为连续张量。

## 给一个新算子声明编译边界：`RMSNormFwdOp`

接入一个算子要写的代码：边界怎么声明、fake 怎么写、target 判定为什么要在节点内部重做一次。其中的追踪、切图、guard 见[dynamo 是怎么工作的](#dynamo)。

`RMSNormFwdOp` 是仓内第一个接入的算子。下面是它的骨架，方法体一律省略，完整代码见 [`src/tileops/ops/norm/rms_norm.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/ops/norm/rms_norm.py)：

```python
class RMSNormFwdOp(Op):
    # 图中属于这个算子的算子名
    compile_op_names = ("tileops::norm_rms_norm_fwd",)

    def _infer_output_shapes(self, x_shape, weight_shape):
        return {"output": tuple(x_shape)}          # manifest 的 shape_rules

    def forward(self, x, weight):
        # 唯一一行：调用那个不透明算子
        return _rms_norm_fwd(x, weight, self._instance_key)

    def _eager_forward(self, x, weight):
        ...                                        # 校验、连续化
        kernel = self.get_or_build_kernel(
            "rms_norm", (x, weight), key=x.dtype, build=...,
        )
        return kernel(x, weight)


@torch.library.custom_op("tileops::norm_rms_norm_fwd", mutates_args=())
def _rms_norm_fwd(x, weight, instance_key: str) -> torch.Tensor:
    return get_instance(instance_key)._eager_forward(x, weight)


@_rms_norm_fwd.register_fake
def _rms_norm_fwd_fake(x, weight, instance_key):
    op = get_instance(instance_key)
    shapes = op._infer_output_shapes(tuple(x.shape), tuple(weight.shape))
    return x.new_empty(shapes["output"])
```

一次调用经过的各层，以及边界落在哪里：

<figure class="callpath" markdown="0">
  <div class="cp-step cp-traced"><code>Op.__call__</code><span>判定 target，失败则撤销</span></div>
  <div class="cp-step cp-traced"><code>forward</code><span>一行，调用不透明算子</span></div>
  <div class="cp-boundary"><span>编译边界</span></div>
  <div class="cp-step cp-opaque"><code>_rms_norm_fwd</code><span>算子体，取回算子实例</span></div>
  <div class="cp-step cp-opaque"><code>_eager_forward</code><span>校验、连续化、取 kernel、launch kernel</span></div>
  <figcaption>紫色两层在 dynamo 的追踪范围内，<code>forward</code> 那一行是它追到的最后一处；界下由不透明算子接手，编译器看不见。</figcaption>
</figure>

其中三处写法不是任选的。

**第一处，算子实例通过字符串键找回，而不是直接传对象。** schema 的类型只有 `Tensor`、`int`、`float`、`bool`、`str` 等固定几种，没有「任意 Python 对象」；而算子体要用的 `kernel_map`、已定下的 target、kernel 记忆表都挂在实例上，摊不成 schema 参数。键还有两个细节不能改：

- **取字符串，不取整数。** 字符串在追踪期是常量，整数会被泛化成 `SymInt`。
- **从不复用。** 正因为是常量，inductor 会把 fake 给出的形状固化进产物；复用键的算子会继承前一个实例的形状。

**第二处，fake 用 `x.new_empty(shape)` 构造，而不是 `torch.empty_like(x)`。** fake 返回的张量，形状、dtype 与 stride 三项都必须与真实执行返回的一致；不一致或在追踪期报错，或在运行期按错误布局访问而静默出错。算子体先连续化再写入新分配的输出，真实输出恒为连续，而 `empty_like` 会把入参的 stride 一起复制 —— 非连续输入就让 fake 宣称了一种真实执行不会产出的布局。

**第三处，target 判定在 `Op.__call__` 与 `get_or_build_kernel` 中各做一次。** 追踪期执行 `self.x = ...`，dynamo 把这次写入记成待办的副作用，等整张图跑完才补上；而不透明节点的执行早于补写，所以节点之外刚写下的判定结果，节点之内读不到。两件事因此都落在节点内部：

- 少了节点内部这一次判定，第一次编译调用会静默用错实现。
- 判定失败时的撤销由做出判定的那一处负责，因为编译产物不保留调用点的 `try/except`。

三处的原因是同一个：torch 的编译与声明机制以函数为单位，而要编译的是一个对象上的一次调用。

## 附录：这条边界为什么是这样

### dynamo 是怎么工作的 {#dynamo}

这一节说明 dynamo 如何决定什么能进入编译图 —— 一个算子要接入 `torch.compile`，需要满足的条件由此而来。

dynamo 是 `torch.compile` 的前端，工作在 CPython 的帧求值层（PEP 523）。

**它的触发入口只有一个：`torch.compile`。** `torch.compile(fn)` 返回一个包装体，包装体被调用时才发生追踪；`nn.Module.compile()` 与装饰器写法是同一入口的另外两种形式。不经过这个入口的调用一律走原来的 Python 路径，与 dynamo 无关 —— 下文把这种路径称为 eager。

第一次调用时，dynamo 接管这一帧，逐条符号执行字节码，把其中的张量运算记成一张 FX 图，把无法进入图的部分留在 Python 里，同时为这张图记下一组 guard，也就是本次追踪所依赖的前提，例如某个张量的 dtype 与维数。此后的调用如果 guard 全部成立，直接复用编译产物；只要有一条不成立，就为新的情况重新追踪一次。

本页用到的三个术语，含义固定如下：

| 术语 | 含义 |
| --- | --- |
| 编译图 | dynamo 捕获下来的那张 FX 图，一次追踪产出一张 |
| 节点 | 图中的一次算子调用，带有输入边以及输出的形状与 dtype |
| 追踪 | 处在 dynamo 的符号执行范围内。追踪期不执行真实计算，只做记录 |

图随后交给后端（inductor 等）完成融合、内存规划与代码生成。图越大，可融合的相邻算子越多，因此算子库的每个算子都要能作为节点出现在使用者的图里。

对接入而言，dynamo 的两条规则是关键：

- **默认一路内联。** 被调用的函数本身不构成边界，函数体会被并入同一次追踪。要让某一段 Python 不被追进去，只能显式声明。
- **追踪不了的代码有两种处理方式。** 默认设置下切图（graph break），把这一段退回 Python 执行，一张图被切成数张；`fullgraph=True` 下直接报错。后者让问题在开发期暴露，因此算子库以 `fullgraph=True` 作为验收条件。

### 算子层与 dynamo 在哪里错位

把上面的规则套到 TileOPs 的算子上，接入的障碍就清楚了：dynamo 编译的单位是帧，也就是函数；而 TileOPs 的算子是对象，一次调用要完成四项工作，其中只有最后一项属于图：

| 一次调用完成的工作 | 是否应当被 dynamo 捕获 |
| --- | --- |
| 校验 dtype 与形状，将输入归一为连续张量 | 不应当 |
| 判定本次调用由哪个 target 服务 | 不应当 |
| 取出或构造 kernel | 不应当，捕获到这里会失败 |
| launch kernel 并得到输出 | 应当，成为编译图中的一个节点 |

这张表需要补充三点。

**「不应当被捕获」不等于「不执行」。** 四项工作在每次调用中都照常发生，区别只在于是否进入编译图。

**这个区分需要人来标注**，dynamo 自己分不出来。torch 为此提供两个接口：`torch.library.custom_op` 把这一次调用注册成一个算子，dynamo 在图里只放一个节点、不追进实现；`register_fake` 告诉编译器这个节点输出什么，它只接收输入的元信息、不接触真实数据。

**不标注就会被追进去，而且一定失败。** 以未声明边界的 `RMSNormFwdOp` 为例，实例的两种状态都编不过去：

- 尚未构造过 kernel 的实例，会在本次调用中现场构造，dynamo 于是追进了构造函数里的 TileLang JIT。
- 已经构造过 kernel 的实例跳过构造，但每次调用仍要重新解析 TileLang program，dynamo 追进 `@tilelang.jit`，停在 `inspect.signature`。

### 为什么边界落在算子层 {#at-op-layer}

边界可以划在算子层，也可以划在更靠下的 kernel 层。两者的差别落在使用者的编译图上。

图中那个节点的身份 —— 名字、参数、粒度以及 fake 给出的输出 —— 就是使用者看到的算子。如果边界划在 kernel 层，换一个后端就换掉了这个节点，同一个算子在不同 target 下会编出不同的图，编译产物于是与后端绑定。划在算子层则不会：节点的身份由算子决定，与哪个后端在服务它无关。

这个位置同时决定了 fake 的写法。算子层并不知道外部 kernel 内部如何分块、如何 padding，唯一对所有 target 共同成立的形状规则写在 manifest 里，因此 fake 只能照 manifest 推导。

节点内部对编译器不可见，但它对外的契约是完整的：schema 给出名字与参数类型，fake 给出输出的形状、dtype、设备与 stride，别名标注说明它不就地修改入参。契约完整，得失也就分得清楚：

- **节点之间的优化照常。** 排布 buffer、计算生命周期、与无依赖的相邻节点交换顺序、无人使用时整体删除。
- **节点内部的优化没有了。** 相邻算子融不进来，输出必须写入显存。

对算子库来说这个取舍是值得的：节点内部是 TileLang 已经编译好的 kernel，本来就不需要 inductor 介入。

### 边界的代价

以下数据在空闲的 H200 上实测，形状为 2048×4096、dtype 为 fp16；每次调用的数字取 2000 次迭代 × 9 轮、三次运行中的最小值：

| | 边界位于 kernel 层 | 边界位于算子层 |
| --- | --- | --- |
| kernel 时间 | 0.0119 ms | 0.0117 ms |
| eager 路径每次调用 | 42.5–45.2 µs | 38.2–42.0 µs |

kernel 本身不受影响，边界位于哪一层与 kernel 如何计算无关。eager 路径快 3–5 µs，原因是边界上移之后一次调用只需穿过一层算子边界。

编译图一侧的代价见[为什么边界落在算子层](#at-op-layer)：融合不跨越节点边界，节点的输出必定写入显存。

### 编译边界不提供的能力

| 不提供 | 原因 |
| --- | --- |
| 跨节点边界的融合 | 节点内部对编译器不可见，两侧的 elementwise 计算只能留在节点之外 |
| 穿过节点的 autograd | 这条调用链服务推理，fwd 与 bwd 各自是独立的算子 |
| 在同一份编译产物中切换 target | target 属于算子实例，更换 target 即更换实例，也就更换了编译图 |
| 用 meta 张量构造 kernel | 传入 meta 张量的调用只回答形状与 dtype |
