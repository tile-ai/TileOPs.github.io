# 如何为 op 新增 kernel

本页说明为已有 op 新增一个 kernel 的两步、什么都不声明时的默认规则、何时新开 kernel 接口，以及如何测试选择结果。选择的整体过程见[首页](index.md)。

## 1. 第一步：把实现登记到 op {#register}

实现是同时继承 `Kernel`（或它的子类）与一个 kernel 接口的类。TileOPs 开发者把它加入 op 的 `kernel_types`，key 使用 snake_case；backend 作者调用 `register_kernel_type`，见 [backend 接入 2](backends.md#register)。

```python
# src/tileops/ops/norm/batch_norm.py
class BatchNormFwdOp(Op):
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "fwd_train_whole": BatchNormFwdTrainWholeKernel,
        "fwd_train_wide": BatchNormFwdTrainWideKernel,
        "fwd_train_split": BatchNormFwdTrainSplitKernel,
        "fwd_train_kernel": BatchNormFwdTrainKernel,
        "fwd_infer_kernel": BatchNormFwdInferKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {
        "batch_norm_fwd_train": BatchNormTrainFwdInterface,
        "batch_norm_fwd_infer": BatchNormInferFwdInterface,
    }
```

一个 key 属于它的类所继承的接口，不需要另写映射。实现类还要满足：

- `forward` 按位置接受接口 `forward` 的全部参数；
- 类方法 `entry_for(call)` 返回 `(build identity, 构建函数)`。build identity 包含一切会改变构建结果的事实，必须可哈希，构建函数只在它第一次出现时运行。默认实现以整个 call spec 为 build identity、以 `cls(call)` 构建；
- 构造函数由实现自行决定，op 从不直接构造实现；
- 调优不经过构建函数：op 处于调优模式时，在调用的设备上对构建出的 kernel 调用 `request_tune()`。

## 2. 第二步：声明实现服务哪些调用 {#rule}

实现的适用范围写在类方法 `refusal(call)` 中：服务这次调用时返回 `None`，否则返回不服务的原因，报错信息列出这个原因。它只描述本实现服务的调用，不描述其他实现。覆写以 `super().refusal(call)` 结尾，使基类声明的限制仍然生效。多个实现共用的判断写成 family call spec 的属性（例如 `AttentionCall.dense_decode_region`），同一继承链内共用的判断写成基类的类方法。

新实现的适用范围与已有实现重叠时，按重叠的对象决定是否声明优先关系：

**表 1** 是否声明优先关系

| No. | 新实现与谁重叠 | 声明 |
| --- | --- | --- |
| 1 | 不与任何实现重叠 | 无 |
| 2 | 只与 general 实现重叠 | 无，general 实现低于其他所有实现 |
| 3 | 与另一个非 general 实现重叠 | 在应当胜出的一方写 `preferred_over = frozenset({"<另一方的 key>"})` |

- `general = True` 表示兜底：服务其他实现都不服务的调用，每个接口至多一个。
- `preferred_over` 只说明两者都适用时谁胜出，不要求一方的适用范围包含在另一方之内。
- 一个实现不在自己的 `refusal` 中排除另一个实现的范围；需要让出时，由另一个实现声明 `preferred_over`。
- 没有声明的重叠在调用时报歧义，不会被按顺序悄悄决定。

GQA dense decode 的三个实现是第 3 种情况。bs1 服务 batch 为 1 的调用，long-context 服务 KV 较长的调用，二者在 batch 为 1 且 KV 较长时重叠，由 long-context 声明它胜出：

```python
# src/tileops/kernels/attention/gqa/decode.py
class GQADecodeLongContextKernel(GQADecodeKernel):
    general: bool = False
    preferred_over = frozenset({"gqa_dense_decode_bs1"})

    @classmethod
    def refusal(cls, call: AttentionCall) -> Optional[str]:
        served = (
            call.dense_decode_region
            and not call.fuse_rope
            and call.seqlen_kv >= 1024
            and call.batch == 1
            and call.heads == 32
            and call.heads_kv == 4
            and call.dim == 128
            and call.dtype == torch.float16
            and call.softcap == 0.0
        )
        return None if served else "does not serve this call"
```

两个非 general 实现的适用范围只要有交集，就需要声明优先关系，不要求一个范围整体包含另一个。FP8 decode 与通用 FP8 实现在一部分调用上重叠，FP8 decode 声明 `preferred_over = frozenset({"gqa_dense_fp8"})`，在交集内胜出；通用 FP8 实现不在自己的 `refusal` 中排除 decode 的范围。

`GQADecodeKernel` 是 general，支持 SM80/89/90；bs1 只支持 SM90。可用性在比较优先关系之前过滤，因此 SM80 上 bs1 不参与选择：

**表 2** batch 为 1 的 decode 在不同架构上的选择

| No. | 架构 | KV 长度 | 可用且适用的实现 | 选中 |
| --- | --- | --- | --- | --- |
| 1 | SM90 | 4096 | decode、bs1、long-context | long-context |
| 2 | SM90 | 512 | decode、bs1 | bs1 |
| 3 | SM80 | 4096 | decode、long-context | long-context |
| 4 | SM80 | 512 | decode | decode |

## 3. 实现什么都不声明时的默认选择规则 {#default}

**表 3** 未声明时的默认值

| No. | 成员 | 默认值 | 效果 |
| --- | --- | --- | --- |
| 1 | `devices` | `frozenset({"cuda"})` | 在 CUDA 设备上可用 |
| 2 | `supported_archs` | `None` | 在所有架构上可用 |
| 3 | `refusal` | 返回 `None` | 服务全部调用 |
| 4 | `general` | `False` | 不低于其他实现 |
| 5 | `preferred_over` | 空集 | 不胜过任何实现 |

只有一个实现的接口因此只需继承接口。`LayerNormKernel` 只声明了支持的架构与自己的 `entry_for`：

```python
# src/tileops/kernels/norm/layer_norm.py
class LayerNormKernel(Kernel, LayerNormFwdInterface):
    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def entry_for(cls, call: LayerNormCall) -> Entry:
        identity = (call.n, call.eps, call.dtype)
        return identity, lambda: cls(*identity)
```

## 4. 新 kernel 何时需要新开 kernel 接口 {#interface}

新 kernel 改变了语义或调用契约，例如返回值不同，就属于一个新的 kernel 接口；只在 shape、dtype、架构或性能上不同，就是已有接口的一个新实现。同一算法换一组 tile 大小或 split 数，属于实现内部的参数，不新增实现。

**表 4** 常见情况的归属

| No. | 情况 | 归属 |
| --- | --- | --- |
| 1 | BatchNorm 的训练与推理：统计量来源不同，训练多返回 `mean` 与 `rstd` | 两个接口 |
| 2 | GLA 的 prefill 与 decode：调用契约相同，只是序列长度不同 | 同一接口的不同实现 |
| 3 | 某个 shape 范围或某个架构上有更快的算法 | 新实现 |
| 4 | 同一算法的 split 数为 1 或大于 1 | 实现内部的参数 |

新接口需要一个 call spec 类型与一个接口类。family 下有多个 kernel 文件时，两者都写在 `src/tileops/kernels/<family>/call_spec.py`；只有一个 kernel 文件时，两者都写在这个 kernel 文件中。接口类按 `{Name}{Fwd|Bwd}Interface` 命名，表示变体的词写在方向后缀之前，例如 `BatchNormTrainFwdInterface`，由 pre-commit 的 `interface-names-lint` 检查。同一 family 的多个接口可以共用一个 call spec 类型，例如 BatchNorm 与 InstanceNorm 的六个接口都用 `BatchNormCall`。

```python
# src/tileops/kernels/norm/call_spec.py
@dataclasses.dataclass(frozen=True)
class LayerNormCall(CallSpec):
    """The facts that select an implementation normalizing trailing rows and build it.

    Layer, RMS, fused-add and adaptive layer normalization take it. ``n`` is the row's
    element count and ``eps`` the op's epsilon.
    """

    n: int = 0
    eps: float = 1e-5
    dtype: torch.dtype = torch.float16

class LayerNormFwdInterface(KernelInterface):
    """Layer normalization over the trailing ``call.n`` elements."""

    request = LayerNormCall

    @abstractmethod
    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Normalize each run of ``call.n`` trailing elements; nothing is written in place.

        Every tensor is contiguous on ``call.device``.

        Args:
            x: Any shape whose trailing axes hold ``call.n`` elements, in ``call.dtype``.
            weight: ``call.n`` elements of scale in ``call.dtype``.
            bias: ``call.n`` elements of shift in ``call.dtype``.

        Returns:
            A new output shaped like *x*, in ``call.dtype``.
        """
```

- call spec 的字段是各实现在 `refusal`、`entry_for` 中读取的调用事实：shape、dtype、语义参数（包括 op 构造时固定的参数）与 `device`。字段只能是不可变的值。不放张量内容、调优策略与优先级；设备事实由 `CallSpec` 提供。
- 接口类的 `request` 指向 call spec 类型；抽象方法 `forward` 的参数表是 op 调用 kernel 时传入的参数，docstring 写明每个张量的 shape、dtype、内存布局、设备、是否被原地写入，以及返回值。backend 只依据这份契约编写实现。
- op 在 `forward` 之外还会在 entry 上调用的方法，一并写成接口的抽象方法。实现必须实现接口的全部抽象方法，否则这个类不能实例化。
- 各实现必须一致的取值，写成接口的普通类方法，由接口算出。例如 op 要在选择之前按大小分配一个输出缓冲区时，这个大小由接口的类方法从 call spec 算出，而不是从已选中的实现上读。
- docstring 写的必须是实现真正做到的：声明输入连续，op 就要在调用前调用 `.contiguous()`；实现不保证的行为（例如输入含 NaN 时的结果）写明未定义，不写成保证。
- 持有 kernel 的 op 都声明 `interfaces`，不覆写 `entry_for`，也不自建 kernel 缓存。

## 5. 如何测试选择结果，以及常见报错 {#tests}

为每个适用范围以及相邻范围的边界各写一个用例，用 `key_for(interface, call)` 检查选中的 key 或拒绝的原因，并显式给出设备事实，使测试不依赖运行它的机器：

```python
# tests/ops/test_family_dispatch.py
def test_gemm_k_too_narrow_to_vectorize_is_refused_during_selection() -> None:
    op = GemmFwdOp()
    call = GemmCall(arch=_SM90, sm_count=132, m=64, n=64, k=1, dtype=torch.float16, trans_b=True)

    with pytest.raises(ValueError, match="k must span at least one"):
        op.key_for("gemm", call)
```

**表 5** 常见报错

| No. | 时刻 | 报错信息的片段 | 原因 |
| --- | --- | --- | --- |
| 1 | 构造 op | `does not implement <Interface>` | key 登记的类继承了接口，但没有继承 `Kernel`；让它同时继承 `Kernel` 与接口，并由 `entry_for(call)` 构建 |
| 2 | 构造 op | `forward does not take <Interface>'s arguments` | `forward` 的参数与接口不符 |
| 3 | 构造 op | `has more than one general implementation` | 一个接口有两个 general 实现 |
| 4 | 构造 op | `preferences form a cycle through` | `preferred_over` 成环 |
| 5 | 构造 op | `keys implement none of its kernel interfaces` | `kernel_types` 中的类没有继承 op 的任何 kernel 接口 |
| 6 | 调用 | `in-tree kernels do not run on` | 没有任何 key 能在调用设备类型上运行（`OpNotAvailableError`） |
| 7 | 调用 | `no implementation serves this call` | 有 key 支持该设备类型，但没有实现同时可用且适用，信息中列出每个实现的理由 |
| 8 | 调用 | `dispatch is ambiguous` | 多个实现都适用，且互相没有优先关系 |
| 9 | 调用 | `takes a <Request> call spec` | 传给 `kernel_for` 的 call spec 类型不对 |
| 10 | 调用 | `cannot key a dispatch cache` | call spec 中有可变的字段，例如 list |
| 11 | 调用 | `this call spec states ['arch']` | 传给 `kernel_for` 的 call spec 显式给出了设备事实；它们应由 `device` 推出，只有 `key_for` 接受显式给出的设备事实 |

新 op 没有声明 `interfaces` 而持有 kernel 时，`tests/test_kernel_dispatch.py` 中的清单测试报 `declare interfaces instead`。
