# 接入 torch.compile

本页说明如何把一个 TileOPs 算子接入 `torch.compile`，也就是让使用者编译自己的模型时，模型中调用的这个算子作为编译图中的一个节点出现，并且这个节点的形态不随服务它的后端而变化。

接入要做的事只有一件：在算子层显式声明一条编译边界，边界之外交给 dynamo 追踪，边界之内对编译器不可见。为什么必须这样做、这条边界为什么落在算子层，取决于 dynamo 的工作方式，因此下文从它开始，再以 `RMSNormFwdOp` 为例走读接入后的代码，最后给出接入之后成立的保证、调用时需要遵守的约定，以及这条边界的代价与限制。

## 背景：dynamo 的工作方式

这一节说明 dynamo 如何决定什么能进入编译图 —— 一个算子要接入 `torch.compile`，需要满足的条件由此而来。

dynamo 是 `torch.compile` 的前端，工作在 CPython 的帧求值层（PEP 523）。

**它的触发入口只有一个：`torch.compile`。** `torch.compile(fn)` 返回一个包装体，包装体被调用时才发生追踪；`nn.Module.compile()` 与装饰器写法是同一入口的另外两种形式。不经过这个入口的调用一律走原来的 Python 路径，与 dynamo 无关 —— 下文把这种路径称为 eager。

第一次调用时，dynamo 接管这一帧，逐条符号执行字节码，把其中的张量运算记成一张 FX 图，把无法进入图的部分留在 Python 里，同时为这张图记下一组 guard，也就是本次追踪所依赖的前提，例如某个张量的 dtype 与维数。此后的调用如果 guard 全部成立，直接复用编译产物；只要有一条不成立，就为新的情况重新追踪一次。

以下三个词在本页中的含义固定：

| 词 | 指什么 |
| --- | --- |
| 编译图 | dynamo 捕获下来的那张 FX 图，一次追踪产出一张 |
| 节点 | 图中的一次算子调用，带有输入边以及输出的形状与 dtype |
| 追踪 | 处在 dynamo 的符号执行范围内。追踪期不执行真实计算，只做记录 |

图随后交给后端（inductor 等）完成融合、内存规划与代码生成。图越大，可融合的相邻算子越多，因此算子库的每个算子都要能作为节点出现在使用者的图里。

对接入而言，dynamo 的两条规则是关键：

- **默认一路内联。** 被调用的函数本身不构成边界，函数体会被并入同一次追踪。要让某一段 Python 不被追进去，只能显式声明。
- **追踪不了的代码有两种下场。** 默认设置下切图（graph break），把这一段退回 Python 执行，一张图被切成数张；`fullgraph=True` 下直接报错。后者让问题在开发期暴露，因此算子库以 `fullgraph=True` 作为验收条件。

## 接入的障碍：算子层与 dynamo 的错位

把上面的规则套到 TileOPs 的算子上，接入的障碍就清楚了：dynamo 编译的单位是帧，也就是函数；而 TileOPs 的算子是对象，一次调用要完成四项工作，其中只有最后一项属于图：

| 一次调用完成的工作 | 是否应当被 dynamo 捕获 |
| --- | --- |
| 校验 dtype 与形状，将输入归一为连续张量 | 不应当 |
| 判定本次调用由哪个 target 服务 | 不应当 |
| 取出或构造 kernel | 不应当，捕获到这里会失败 |
| 发射 kernel 并得到输出 | 应当，成为编译图中的一个节点 |

三点需要说明。

**「不应当被捕获」不等于「不执行」。** 四项工作在每次调用中都照常发生，区别只在于是否进入编译图。

**这个区分需要人来标注**，dynamo 自己分不出来。torch 为此提供两个接口：`torch.library.custom_op` 把这一次调用注册成一个算子，dynamo 在图里只放一个节点、不追进实现；`register_fake` 告诉编译器这个节点输出什么，它只接收输入的元信息、不接触真实数据。

**不标注就会被追进去，并且必然失败。** 以未声明边界的 `RMSNormFwdOp` 为例，两种状态都无法编译：尚未构造过 kernel 的实例会在本次调用中现场构造，dynamo 追进构造函数里的 TileLang JIT；已经构造过的实例虽然跳过构造，但每次仍要重新解析 TileLang program，dynamo 追进 `@tilelang.jit` 后停在 `inspect.signature`。

## 为什么编译边界需要位于算子层？

边界可以划在算子层，也可以划在更靠下的 kernel 层。两者的差别落在使用者的编译图上。

图中那个节点的身份 —— 名字、参数、粒度以及 fake 给出的输出 —— 就是使用者看到的算子。如果边界划在 kernel 层，换一个后端就换掉了这个节点，同一个算子在不同 target 下会编出不同的图，编译产物于是与后端绑定。划在算子层则不会：节点的身份由算子决定，与哪个后端在服务它无关。

这个位置同时决定了 fake 的写法。算子层并不知道外部 kernel 内部如何分块、如何 padding，唯一对所有 target 共同成立的形状规则写在 manifest 里，因此 fake 只能照 manifest 推导。

节点内部对编译器不可见，但契约是完整的：schema 给出名字与参数类型，fake 给出输出的形状、dtype、设备与 stride，别名标注说明它不就地修改入参。因此**节点之间**的优化照常 —— 排布 buffer、计算生命周期、与无依赖的相邻节点交换顺序、无人使用时整体删除；失去的是**节点内部**的优化，相邻算子融不进来，输出必须写入显存。这个取舍对算子库是成立的：节点内部是 TileLang 已经编译好的 kernel，本来就不需要 inductor 介入。

## 接入后的代码：`RMSNormFwdOp`

这一节把前面几节的结论落到代码上：怎么声明边界、fake 怎么写、target 判定为什么要在节点内部重做一次。

`RMSNormFwdOp` 是仓内第一个完成接入的算子，编译边界放在算子层。下面是它的骨架，方法体一律省略，完整代码见 [`src/tileops/ops/norm/rms_norm.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/ops/norm/rms_norm.py)：

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

一次调用经过的代码，以及每一层是否被追踪：

```
Op.__call__        判定 target，失败则撤销          —— 被追踪
  forward          一行，调用不透明算子            —— 被追踪，追到这里为止
    _rms_norm_fwd  算子体，取回算子实例            —— 不被追踪
      _eager_forward  校验、连续化、取 kernel、发射 —— 不被追踪
```

这段代码里有三处写法不是任意选择的，各有其原因。

**第一处，算子实例通过一个字符串键找回，而不是直接传对象。** schema 的类型系统只有 `Tensor`、`int`、`float`、`bool`、`str` 等固定几种，没有「任意 Python 对象」，而算子体要用到的 `kernel_map`、已定下的 target、kernel 记忆表都挂在实例上，摊不成 schema 参数。键取字符串而非整数，且从不复用：字符串在追踪期是编译期常量，整数会被泛化成 `SymInt`；而正因为它是常量，inductor 会把 fake 给出的形状烤进产物，复用键的算子会继承前一个实例的形状。

**第二处，fake 用 `x.new_empty(shape)` 构造，而不是 `torch.empty_like(x)`。** fake 返回的张量在形状、dtype 与 stride 三项上都必须等于真实执行返回的那个，不一致或在追踪期报错，或在运行期按错误布局访问而静默出错。算子体先做连续化再写入新分配的输出，真实输出恒为连续；`empty_like` 会连入参的 stride 一起复制，非连续输入就会让 fake 宣称一种真实执行不会产出的布局。

**第三处，target 判定在 `Op.__call__` 与 `get_or_build_kernel` 中各做一次。** 追踪期执行 `self.x = ...` 时，dynamo 把这次写入记成待办的副作用，等整张图跑完才补上，而不透明节点的执行早于补写 —— 节点之外刚写下的判定结果，节点之内读不到。少了节点内部这一次判定，第一次编译调用会静默用错实现；同理，判定失败时的撤销也必须由做出判定的那一处负责，因为编译产物不保留调用点的 `try/except`。

三处写法同出一源：torch 的编译与声明机制都以函数为单位，而需要编译的是一个对象上的一次调用。

## 三项保证

边界的位置确定之后，使用者可以依赖以下三点。

- **编译图不随 target 变化。** 更换后端或更换硬件，同一段代码编出的图完全相同，编译产物因此与后端无关。
- **`fullgraph=True` 可用。** 前提是该算子已经声明这条契约，判定方法见下一节。
- **输出的形状、dtype 与 stride 由 manifest 规定。** 它们与 kernel 内部如何分块、如何 padding 无关。输入在节点内部完成连续化，输出恒为连续张量。

使用方式没有额外步骤，构造算子实例并把调用它的函数交给 `torch.compile` 即可：

```python
import torch
from tileops.ops import RMSNormFwdOp

op = RMSNormFwdOp(normalized_shape=(4096,))     # 构造一次，反复使用

@torch.compile(fullgraph=True)
def block(x, weight):
    return op(x, weight)

x = torch.randn(2048, 4096, device="cuda", dtype=torch.float16)
w = torch.randn(4096, device="cuda", dtype=torch.float16)
block(x, w)
```

以 `TORCH_LOGS=graph_code` 运行即可输出捕获到的图：其中出现的是 `tileops::norm_rms_norm_fwd` 这一个节点，而不是 kernel 内部的多次调用。

## 已支持的算子

判定方法是读取类属性 `compile_op_names`。取值非空，说明该算子的编译边界已经位于算子层，`fullgraph=True` 可用；取值为空 tuple，说明它尚未迁移。

```python
>>> from tileops.ops import RMSNormFwdOp
>>> RMSNormFwdOp.compile_op_names
('tileops::norm_rms_norm_fwd',)
```

尚未迁移的算子在 `fullgraph=True` 下直接报错，在默认设置下则发生切图。

## 调用约定

以下五条约定各自对应上文的一处机制，违反其中任何一条都会让编译路径的行为不同于 eager 路径。

- **算子实例构造一次并反复使用。** 实例键是编译期常量，每个实例对应一张独立的编译图；在循环内部新建实例意味着每次迭代都重新编译。
- **不要依赖 stride 原样传递。** 非连续输入在节点内部完成连续化，输出恒为连续张量；如果后续计算需要其他布局，应在算子之外自行转换。
- **不能用 meta 张量预热。** 算子一旦有了边界，传入 meta 或 fake 张量的调用就在 fake 处返回，不会执行到构造 kernel 的那一步。
- **CUDA graph 捕获之前先行预热。** 用真实张量、相同形状至少调用一次：构造 kernel 允许编译，而捕获期间只允许查表命中后直接调用。各阶段分别允许执行哪些操作，见[接入新硬件后端](backends.md)。
- **每个设备各自构造 kernel。** 设备是 kernel 记忆键的一部分，同一个算子实例在第二块卡上会重新构造一次。构造函数中指名的 `target=` 在首次编译调用中同样生效；构造失败不会把该算子固定到任何 target。

## 边界的代价

以下数据在空闲的 H200 上实测，形状为 2048×4096、dtype 为 fp16；每次调用的数字取 2000 次迭代 × 9 轮、三次运行中的最小值：

| | 边界位于 kernel 层 | 边界位于算子层 |
| --- | --- | --- |
| kernel 时间 | 0.0119 ms | 0.0117 ms |
| eager 路径每次调用 | 42.5–45.2 µs | 38.2–42.0 µs |

kernel 本身不受影响，边界位于哪一层与 kernel 如何计算无关。eager 路径快 3–5 µs，原因是边界上移之后一次调用只需穿过一层算子边界。

编译图一侧的代价已在前文说明：融合不跨越节点边界，节点的输出必定写入显存。

## 编译边界不提供的能力

| 不提供 | 原因 |
| --- | --- |
| 跨节点边界的融合 | 节点内部对编译器不可见，两侧的 elementwise 计算只能留在节点之外 |
| 穿过节点的 autograd | 这条调用链服务推理，fwd 与 bwd 各自是独立的算子 |
| 在同一份编译产物中切换 target | target 属于算子实例，更换 target 即更换实例，也就更换了编译图 |
| 用 meta 张量构造 kernel | 传入 meta 张量的调用只回答形状与 dtype |
