# 组合 op

组合 op 通过子 op 完成计算，不构造或只部分构造自己的 kernel。本页说明：

- `Op` 基类如何声明和持有子 op；
- 子 op 的调用如何归入父调用；
- 子 op 在何时构造。

## 1. 声明与持有 {#declare}

`MoEExpertMLPFwdOp` 由两个 grouped GEMM 组成：

```python
class MoEExpertMLPFwdOp(Op):
    delegate_types: ClassVar[Mapping[str, type[Op]]] = {
        "gate_up": MoEGroupedGemmFwdOp,
        "down": MoEGroupedGemmFwdOp,
    }

    def __init__(self, layout, activation="silu_and_mul", *, target=None):
        """Configure two grouped GEMMs on ``layout``, the first fusing the gated activation."""
        self.layout = layout
        self.activation = activation
        super().__init__(target=target)
        self.gate_up = self.delegate_for("gate_up", None, layout=layout, activation=activation)
        self.down = self.delegate_for("down", None, layout=layout)
```

示例省略了类 docstring 和 `forward`，完整代码见 [`src/tileops/ops/moe/staged.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/ops/moe/staged.py)。子 op 遵循与 kernel 相同的规则：先在类上声明，再经基类的方法取得。

- **声明。** `delegate_types` 按 **stage** 名映射组合 op 可能持有的子 op 类；validator 检查它与 manifest 条目的 composition 一致。
    - 映射的顺序就是 stage 的顺序，与 manifest 的 composition 一致。
    - 组合关系因此是类的事实，在任何调用之前就能检查。
- **持有。** `delegate_for(stage, identity, given=None, /, **params)` 是持有子 op 的唯一方式。
    - 同一个 `(stage, identity)` 只构造一次。
    - `identity` 包含所有会改变子 op 构造结果的参数，与 kernel 侧 `entry_for` 返回的 identity 同义。
    - 构造参数在组合 op 构造时就已确定时，`identity` 取 `None`。
- **继承执行策略。**
    - 基类构造的子 op 继承组合 op 的 `target`。
    - 调用方注入的子 op 通过 `given` 传入，按原样持有，保持自己的 `target`。
    - 组合 op 处于调优模式时，基类对新持有的子 op 调用 `request_tune()`，构造的与注入的都是如此。
- **一个实例只登记一次。**
    - 同一个 `(stage, identity)` 的重复调用，返回已持有的实例。
    - 把已经持有的实例用另一个 `(stage, identity)` 再次登记，`delegate_for` 报错。
- **派生的枚举。** `held_delegates()`、`iter_kernels()` 和 `request_tune()` 从 `delegate_for` 持有的子 op 派生，组合 op 不覆盖它们。

## 2. 子 op 的调用归入 stage {#stages}

子 op 的调用完成时：

1. 子 op 把记录报告给调用栈上的父调用，即[构造与调用 § 一次调用的七步](lifecycle.md#serve)的第 7 步；
1. 父调用按 `delegate_for` 登记的 stage，把它归入自己记录的 `stages`。

完成调用的子 op 未经 `delegate_for` 持有时，基类报错，父调用失败。因此：

- `stages` 要么完整，要么调用失败；
- 依赖 `stages` 的 roofline 不会静默算错。

## 3. 子 op 的构造时机 {#when}

**表 1** 子 op 的构造时机

| No. | 子 op 的构造参数 | 在哪里调用 `delegate_for` | 编译 |
| --- | --- | --- | --- |
| 1 | 构造时已确定 | 构造函数中，`super().__init__` 之后 | 子 op 在 trace 之前已持有 |
| 2 | 调用时才确定 | `forward` 中 | 不承诺冷启动的 `fullgraph=True` 编译 |

表 1 第二行的原因：

- 构造子 op 会运行子 op 的 `Op.__init__`，dynamo 无法 trace。
- 这样的组合 op 只在这次调用所需的子 op 都已持有之后，才能被 trace。
- 被 trace 时，组合 op 本身不是图中的节点。没有 target 的 builder、`forward` 调用子 op 且子 op 有编译边界时，子 op 的 custom op 各自是图中的节点。

`FusedMoEExpertsFwdOp` 属于第二行：它在 `forward` 中按这次调用的专家数持有 `pre_permute` 子 op，一个实例服务多个专家数。

## 4. target 与组合 op {#target}

- **选定的 target 没有为组合 op 注册 builder 时：**
    - 组合 op 不声明自己的 kernel，则运行自己的组合；
    - 组合 op 也声明自己的 kernel，则调用报 `OpNotAvailableError`。
- **子 op 的 target：** 基类构造的子 op 继承组合 op 的 `target`；通过 `given` 注入的子 op 保持自己的 `target`。每个子 op 各自选择服务的 target。
- **撤销：** 组合 op 的 target 选择在调用中被撤销时，基类对持有的子 op 递归撤销，见[构造与调用 § 失败与撤销](lifecycle.md#failure)。
