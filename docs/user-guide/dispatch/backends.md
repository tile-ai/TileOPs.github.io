# How a backend joins TileOPs

A third-party backend has two ways to join, ordered from the smaller range of calls taken over to the larger: add an implementation to a kernel interface, or replace a whole op. In the first, the backend's class and the in-tree implementations follow the same kernel interface; how to write one is in [Adding a kernel to an op](writing.md).

## 1. Choose the way by the range of calls to take over {#choose}

![The ranges taken over by the two ways of extending](img/extension.svg)

**Figure 1** The range each of the two ways of extending takes over. Teal marks the parts provided by the system, purple the parts written by TileOPs developers, and green the parts provided by the backend.

**Table 1** Comparison of the two ways

| No. | Item | `register_kernel_type` | target |
| --- | --- | --- | --- |
| 1 | What changes | a new key, with its own applicability and precedence | every call of the op, except calls whose written tensors are all empty |
| 2 | Applies to | every instance of the op constructed after registration | the op instances that select the target |
| 3 | Contract followed | the kernel interface | the op's signature in the manifest |
| 4 | Calls the new class does not serve | still served by the in-tree implementations | none; the target serves every call, except calls whose written tensors are all empty |

## 2. Add an implementation: register_kernel_type {#register}

`tileops.backend.register_kernel_type(op, key, kernel_type)` adds an implementation to an op. It is the backend form of step 1 in [Adding a kernel to an op](writing.md). `op` is the op's class name, `key` is the new implementation's name, and `kernel_type` is a class that subclasses both `Kernel` and one of the op's kernel interfaces; the inheritance decides which interface it belongs to, and the class is built through its own class method `entry_for(call)`. Step 2 is the same as for in-tree implementations.

```python
# tests/test_kernel_dispatch.py
class _TorchLayerNorm(Kernel, LayerNormFwdInterface):
    """A replacement written against ``LayerNormFwdInterface`` alone."""

    devices = frozenset({torch.device(run_device()).type})

    def __init__(self, n: int, eps: float) -> None:
        super().__init__()
        self.n, self.eps = n, eps

    @classmethod
    def entry_for(cls, call: LayerNormCall):
        return (call.n, call.eps), lambda: cls(call.n, call.eps)

    def forward(self, x, weight, bias):
        return F.layer_norm(x.float(), (self.n,), weight.float(), bias.float(), self.eps).to(
            x.dtype
        )


class _NarrowTorchLayerNorm(_TorchLayerNorm):
    """An added implementation for short rows, which wins over the in-tree one there."""

    preferred_over = frozenset({"layer_norm"})

    @classmethod
    def refusal(cls, call: LayerNormCall) -> "str | None":
        return None if call.n <= 256 else "serves rows of at most 256"

register_kernel_type("LayerNormFwdOp", "torch_short_rows", _NarrowTorchLayerNorm)
```

- The new implementation overlaps `LayerNormKernel` on `n <= 256`, and `LayerNormKernel` is not general, so the new implementation declares `preferred_over`.
- Calls the new implementation does not serve, such as `n = 1024`, are still served by the in-tree implementation.
- The new implementation enters only op instances constructed after registration.
- Registering the same key twice under one op raises `BackendError`. A key equal to an in-tree key raises `reuse keys it has` when the op is constructed.
- When the new implementation inherits none of the op's kernel interfaces, states `entry_for` other than as a class method, or has a `forward` that does not take the interface's arguments, constructing the op raises; see [Adding a kernel to an op § 5](writing.md#tests).

Registration happens when the backend module is imported. The backend declares an entry point in `pyproject.toml`, and TileOPs imports it when the first op is constructed:

```toml
[project.entry-points."tileops.backends"]
acme = "tileops_acme"
```

When the module fails to import, all its registrations are undone and TileOPs emits a `RuntimeWarning`. The failure records are available through `tileops.backend.load_failures()`.

## 3. Replace a whole op: target {#target}

The full target protocol, a runnable template backend, and common errors are in [Adding a hardware backend](../../backends.md). This section only summarizes how a target differs from the other way.

A target is a name a backend gives to a set of kernels. Once a target registers a builder for an op, every call of an op instance that selects the target is served by the target, and the op's own `forward` does not run. The exception is a call whose written tensors are all empty: then neither the target nor the in-tree implementation runs, and the outputs are constructed from the signature.

```python
from tileops.backend import TensorSpec, register_detector, register_kernel_builder
from .kernels import AcmeRMSNorm

register_detector(target="acme", detect=lambda device: device.type == "acme")

def build_rms_norm(x: TensorSpec, weight: TensorSpec | None, *, normalized_shape, eps):
    return AcmeRMSNorm(normalized_shape, eps, x.dtype)

register_kernel_builder(op="RMSNormFwdOp", target="acme", build_kernel=build_rms_norm)
```

- `build_kernel` receives a `TensorSpec` for each input in the order of the op's `signature.inputs` (an optional input that is not passed is `None`, such as `RMSNormFwdOp`'s `weight`), and receives `signature.params` as keywords. The kernel it returns receives the input tensors in the same order at call time, and receives the output buffers and execution arguments the caller passes as keywords. Importing the module compiles nothing; building happens when `build_kernel` is called.
- An op instance caches the kernels returned by `build_kernel` by the call's device, and by whether each input is passed, its dtype, and its shape.
- An op instance fixes its target on the first call that needs to run an implementation, and the target does not change afterwards. The construction argument `target=` comes first, then the process default set by `set_default_target`, and finally each target's detector inspects the call's device. When no target claims the device, the in-tree implementation is used; when several targets claim it, the call raises `AmbiguousTargetError`. A call whose written tensors are all empty does not fix the target. When the call that fixes the target fails, the choice is undone and the next call fixes it again.
- A call passed to a target has already passed the checks generated from the signature. Every tensor is on the call's device, except tensors declared `device: cpu`. Every input that is not written is contiguous, and a written input is contiguous when it declares `contiguous: true`.
- When the selected target has registered no builder for this op and the op holds kernels of its own, the call raises `OpNotAvailableError` and does not fall back to the in-tree implementation.
