# Adding a kernel to an op

This page describes the two steps of adding a kernel to an existing op, the default rules when nothing is declared, when to open a new kernel interface, and how to test the selection. The overall selection process is in [How an op selects a kernel](index.md).

## 1. Step 1: register the implementation with the op {#register}

An implementation is a class that subclasses both `Kernel` (or one of its subclasses) and a kernel interface. A TileOPs developer adds it to the op's `kernel_types` under a snake_case key. A backend author calls `register_implementation`; see [How a backend joins TileOPs § 3](backends.md#register).

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

A key belongs to the interface its class subclasses; no separate mapping is written. The implementation class also satisfies the following:

- `forward` accepts all parameters of the interface's `forward`, positionally;
- the class method `entry_for(call)` returns `(build identity, factory)`. The build identity contains every fact that changes the build result and must be hashable. The factory runs only the first time that identity appears. The default implementation uses the whole call spec as the build identity and builds with `cls(call)`;
- the constructor is the implementation's own choice; the op never constructs an implementation directly;
- tuning does not go through the factory; the op applies it to the built kernel on the call's device.

## 2. Step 2: declare which calls the implementation serves {#rule}

An implementation's applicability is written in the class method `applies(call)`. It describes only the calls this implementation serves, not other implementations. When an error message needs to state the reason for a refusal, override `refusal(call)`: it returns `None` when the implementation applies and the reason otherwise, and `applies` then returns `cls.refusal(call) is None`. A check shared by several implementations is written as a property of the family's call spec (for example `AttentionCall.dense_decode_region`); a check shared within one inheritance chain is written as a class method of the base class.

When a new implementation's applicability overlaps an existing implementation, what it overlaps decides whether precedence is declared:

**Table 1** Whether to declare precedence

| No. | The new implementation overlaps | Declaration |
| --- | --- | --- |
| 1 | no implementation | none |
| 2 | only the general implementation | none; the general implementation ranks below every other implementation |
| 3 | another non-general implementation | the side that should win writes `preferred_over = frozenset({"<the other key>"})` |

- `general = True` marks the fallback: it serves the calls no other implementation serves. Each interface has at most one.
- `preferred_over` states only which side wins when both apply. It does not require one side's applicability to be contained in the other's.
- An implementation does not exclude another implementation's range in its own `applies`. When it should yield, the other implementation declares `preferred_over`.
- An undeclared overlap raises an ambiguity error at call time; it is never decided silently by order.

The three GQA dense decode implementations are case 3. bs1 serves calls with batch 1, and long-context serves calls with a long KV. The two overlap when batch is 1 and KV is long, and long-context declares that it wins:

```python
# src/tileops/kernels/attention/gqa_decode.py
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

Two non-general implementations need a precedence declaration as soon as their applicability intersects; one range does not have to contain the other. FP8 decode and the generic FP8 implementation overlap on part of the calls. FP8 decode declares `preferred_over = frozenset({"gqa_dense_fp8"})` and wins inside the intersection; the generic FP8 implementation does not exclude the decode range in its own `applies`.

`GQADecodeKernel` is general and supports SM80/89/90; bs1 supports only SM90. Availability filters before precedence is compared, so on SM80 bs1 takes no part in selection:

**Table 2** Selection for batch-1 decode on different architectures

| No. | Architecture | KV length | Available and applicable implementations | Selected |
| --- | --- | --- | --- | --- |
| 1 | SM90 | 4096 | decode, bs1, long-context | long-context |
| 2 | SM90 | 512 | decode, bs1 | bs1 |
| 3 | SM80 | 4096 | decode, long-context | long-context |
| 4 | SM80 | 512 | decode | decode |

## 3. Default selection rules when an implementation declares nothing {#default}

**Table 3** Defaults when nothing is declared

| No. | Member | Default | Effect |
| --- | --- | --- | --- |
| 1 | `devices` | `frozenset({"cuda"})` | available on CUDA devices |
| 2 | `supported_archs` | `None` | available on every architecture |
| 3 | `applies` | returns `True` | serves every call |
| 4 | `general` | `False` | does not rank below other implementations |
| 5 | `preferred_over` | empty set | wins over no implementation |

An interface with a single implementation therefore only needs the class to subclass the interface. `LayerNormKernel` declares only its supported architectures and its own `entry_for`:

```python
# src/tileops/kernels/norm/layer_norm.py
class LayerNormKernel(Kernel, LayerNormFwdInterface):
    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def entry_for(cls, call: LayerNormCall) -> Entry:
        identity = (call.n, call.eps, call.dtype)
        return identity, lambda: cls(*identity)
```

## 4. When a new kernel needs a new kernel interface {#interface}

A new kernel that changes the semantics or the call contract, for example by returning different values, belongs to a new kernel interface. A new kernel that differs only in shape, dtype, architecture, or performance is a new implementation of an existing interface. The same algorithm with a different set of tile sizes or split counts is a parameter inside one implementation, not a new implementation.

**Table 4** Where common cases belong

| No. | Case | Belongs to |
| --- | --- | --- |
| 1 | BatchNorm training and inference: the statistics come from different sources, and training also returns `mean` and `rstd` | two interfaces |
| 2 | GLA prefill and decode: the call contract is the same, only the sequence length differs | different implementations of one interface |
| 3 | a faster algorithm on some shape range or some architecture | a new implementation |
| 4 | the same algorithm with a split count of 1 or greater than 1 | a parameter inside the implementation |

A new interface needs a call spec type and an interface class. When the family has several kernel files, both are written in `src/tileops/kernels/<family>/call_spec.py`; when it has a single kernel file, both are written in that kernel file. Interface classes are named `{Name}{Fwd|Bwd}Interface`, with the word for the variant placed before the direction suffix, for example `BatchNormTrainFwdInterface`; the pre-commit hook `interface-names-lint` checks this. Several interfaces of one family can share a call spec type; for example, the six interfaces of BatchNorm and InstanceNorm all use `BatchNormCall`.

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

- The call spec's fields are the call facts that implementations read in `applies`, `refusal`, and `entry_for`: shapes, dtypes, semantic parameters (including those fixed when the op is constructed), and `device`. Fields hold only immutable values. They do not hold tensor contents, tuning policy, or priorities; device facts are provided by `CallSpec`.
- The interface class's `request` points to the call spec type. The parameter list of the abstract method `forward` is the arguments the op passes when it calls the kernel. Its docstring states, for each tensor, the shape, dtype, memory layout, device, and whether it is written in place, and it states the return value. A backend writes its implementation from this contract alone.
- Methods other than `forward` that the op calls on the entry are also written as abstract methods of the interface. An implementation must implement every abstract method of the interface; otherwise the class cannot be instantiated.
- Values that every implementation must agree on are written as ordinary class methods of the interface and computed by the interface. For example, when the op allocates an output buffer by size before selection, a class method of the interface computes that size from the call spec; the size is not read from the selected implementation.
- The docstring states only what the implementation actually does. If it declares the inputs contiguous, the op calls `.contiguous()` before the call. Behavior the implementation does not guarantee (for example, the result when an input contains NaN) is stated as undefined, not as a guarantee.
- Every op that holds kernels declares `interfaces`, does not override `entry_for`, and does not keep its own kernel cache.

## 5. How to test the selection, and common errors {#tests}

Write one test case for each applicability range and for each boundary between adjacent ranges. Check the selected key with `select_implementation`, and give the device facts explicitly so that the test does not depend on the machine that runs it:

```python
# tests/ops/test_batch_norm.py
def test_each_region_selects_its_one_implementation(
    op_cls, interface, n, c, spatial, dtype, key
) -> None:
    """Exactly one non-general implementation, or else the general one, serves each shape."""
    op = op_cls()
    call = BatchNormCall(arch=90, sm_count=132, n=n, c=c, spatial=spatial, dtype=dtype)
    assert op.select_implementation(interface, call) == key
```

**Table 5** Common errors

| No. | When | Error message fragment | Cause |
| --- | --- | --- | --- |
| 1 | constructing the op | `does not implement <Interface>` | the class that runs behind a key (including a `kernel_map=` replacement) does not subclass the interface; make it subclass the interface and build through `entry_for(call)` |
| 2 | constructing the op | `forward does not take <Interface>'s arguments` | `forward`'s parameters do not match the interface |
| 3 | constructing the op | `has more than one general implementation` | an interface has two general implementations |
| 4 | constructing the op | `preferences form a cycle through` | `preferred_over` forms a cycle |
| 5 | constructing the op | `keys implement none of its kernel interfaces` | a class in `kernel_types` subclasses none of the op's kernel interfaces |
| 6 | calling | `in-tree kernels do not run on` | no key can run on the call's device type (`OpNotAvailableError`) |
| 7 | calling | `no implementation serves this call` | some key supports the device type, but no implementation is both available and applicable; the message lists each implementation's reason |
| 8 | calling | `dispatch is ambiguous` | several implementations apply, and none has precedence over the others |
| 9 | calling | `takes a <Request> call spec` | the call spec passed to `kernel_for` has the wrong type |
| 10 | calling | `cannot key a dispatch cache` | the call spec has a mutable field, such as a list |
| 11 | calling | `this call spec states ['arch']` | the call spec passed to `kernel_for` gives device facts explicitly; they are derived from `device`, and only `select_implementation` accepts explicit device facts |

When a new op holds kernels without declaring `interfaces`, the inventory test in `tests/test_kernel_dispatch.py` raises `declare interfaces instead`.
