# Adding a Backend

A backend is an installable Python package that takes over TileOPs ops on its own
devices. It supplies the kernels; TileOPs keeps the op layer — manifest validation,
parameter normalisation, memoisation, roofline, tests. Adding one requires no change
to TileOPs itself.

[`tileops-backend-example`](https://github.com/lcy-seso/tileops-backend-example) is the
reference an external backend author copies: a package that implements its kernels in
pure PyTorch and claims CPU. Apart from the kernels touching no dedicated hardware,
every other part — entry point, registration, the `build_kernel` signature, the
memoisation rule, the error messages — is what a backend for dedicated hardware writes.

What installing it changes:

```console
$ python -c "import torch; from tileops.ops.norm.rms_norm import RMSNormFwdOp; \
             RMSNormFwdOp(normalized_shape=(64,))(torch.randn(4,64,dtype=torch.float16), \
                                                  torch.randn(64,dtype=torch.float16))"
ValueError: RMSNormKernel is a CUDA kernel; got x on cpu and weight on cpu.

$ pip install -e .

$ python -c "...the same code..."
# returns normally, bit-identical to torch.nn.functional.rms_norm
```

## Minimal implementation

Everything a backend has to provide. Three lines in `pyproject.toml`:

```toml
[project.entry-points."tileops.backends"]
torch_cpu = "tileops_cpu"
```

And two registrations at module top level:

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

`pip install` is the whole activation step: TileOPs enumerates that entry point group
when the first Op is constructed, imports the declared module, and the two top-level
calls fill the registry. There is no other initialisation, no base class to subclass,
no interface to implement.

## Target name vs device type

Two names appear in that registration, and they mean different things:

| Name | Meaning | Decided by |
| --- | --- | --- |
| `target="torch_cpu"` | the name of this set of kernels | the backend author |
| `device.type == "cpu"` | which devices it claims | torch |

They are not in one-to-one correspondence: one device type can carry more than one set
of kernels from different vendors; some hardware arrives through `privateuseone`, whose
string carries no vendor information at all; other backends have to read an environment
variable or call a vendor runtime to decide. So TileOPs does not parse `torch.device` —
it passes it through to `detect` and lets the backend answer.

`detect` answers only which devices belong to the backend. **Whether a particular call
is supported — dtype, shape, parameter combination — is answered by `build_kernel`**,
the only place that sees them. Return `False` for a device you do not claim; do not
raise.

## Example layout

| File | Contents |
| --- | --- |
| `pyproject.toml` | the entry point declaration, which is the whole install mechanism |
| `src/tileops_cpu/__init__.py` | all registration code |
| `src/tileops_cpu/kernels.py` | the kernel implementation. A real backend compiles here |
| `src/tileops_cpu/pending.py` | a registered builder that cannot be called yet — see [below](#op-states-after-install) |
| `tests/test_takeover.py` | numerics, validation, normalisation, outputs |
| `tests/test_discovery.py` | entry point and registration |
| `tests/test_errors.py` | the four error paths |
| `tests/test_memoization.py` | when `build_kernel` is called again |

## Builder signature

A builder receives descriptions of the inputs and returns something callable. Its
signature comes from the op's manifest, its return value is called with the real
tensors, and TileOPs decides when to call it again.

**Writing a kernel needs the manifest, not the TileOPs source.**

`RMSNormFwdOp` in `src/tileops/manifest/normalization.yaml`:

```yaml
signature:
  inputs:                       # declaration order is call order
    x: {dtype: "float16 | bfloat16"}
    weight: {dtype: "same_as(x)"}
  params:                       # passed as keyword arguments under these names
    normalized_shape: {type: "list[int] | tuple[int, ...]"}
    eps: {type: "float | None", default: null}
```

The corresponding builder:

```python
def build_rms_norm(x: TensorSpec, weight: TensorSpec, *, normalized_shape, eps):
```

Two things to note:

- **`eps` arrives as `1e-6`, not `None`.** The manifest default is null, but the op layer
  has already normalised it to a definite value. Every optional parameter behaves this way.
- **`TensorSpec` is a description, not a tensor.** It carries `device` / `dtype` / `shape`
  and nothing else — no data, no reference to a tensor. Two classes of bug become
  unwritable: choosing a kernel by tensor contents (the memo table is indexed by shape, so
  later calls would fetch the wrong kernel), and keeping a tensor alive for the process
  lifetime inside a cached kernel.

The return value has one requirement: **callable**. It receives the real tensors in the
same order, and returns what `signature.outputs` declares — a tensor for a single output,
a tuple in order for several, `None` for a pure in-place write.

**The constructor takes compile-time parameters only.** Values that get compiled into
generated code — tile sizes, dimensions treated as constants, dtypes — belong in the
constructor; the rest belongs to `__call__`. Decode makes this a hard requirement:
`seq_len` grows step by step and batch changes with the running set, so putting them in
the constructor means recompiling every step. That is why `CpuRMSNorm` in `kernels.py`
does not receive the row count when it is constructed.

## The op layer's work

The following is identical for every target, and a backend **must not reimplement it**:

- manifest dtype validation and shape rules
- parameter normalisation (optional values resolved to definite ones)
- input contiguity — every tensor handed to a kernel is contiguous
- memoisation and reuse of kernels
- roofline, profile, numerical tests

**The op layer does not change shapes before handing tensors over.** A kernel receives the
shapes the manifest declares; whatever layout it needs, it arranges itself, inside its own
call wrapper.

**Where the code and the manifest both describe something, the manifest governs.** Output
dtype, shape rules and parameter types are the manifest's; a kernel does not rewrite them,
and reports an error where it cannot comply. The error has to name which item is unmet —
dtype, shape, arch, no implementation available, compilation failed — and the value it
actually received. "Unsupported" on its own is not a diagnosis.

## When a kernel is rebuilt

TileOPs remembers the return value of `build_kernel` by **input signature**:

> `(dtype, shape)` per input, in `signature.inputs` order.

That is: **two calls with the same input signature get the same kernel.**

`device` is not part of the key — whether the artefact differs per device is the backend's
own business, and the callable receives the real tensors every time, so it can dispatch
then. `params` are not part of the key either; they are fixed for an op instance.

A finer distinction is made inside the backend; a coarser grain, to rebuild less often, is
an extra cache inside `build_kernel`. Both directions are resolved on the backend side.

`tests/test_memoization.py` checks this rule item by item, including the key's actual shape:

```python
assert key == ((torch.float16, (4, 64)), (torch.float16, (64,)))  # x first, weight second
```

## Op states after install

Once `detect` claims a device type, **every** op on those devices is served by that
target. A missing one is an error; there is **no fall back to the in-tree
implementation**. The reason is direct: selecting a target means the device belongs to
other hardware, where the kernels TileOPs ships cannot launch, and falling back would
trade a clear "this target does not implement this op" for an incomprehensible launch
failure.

| State | Example | Result |
| --- | --- | --- |
| A builder is registered, and the op passes its tensors to the call site | `RMSNormFwdOp` | runs |
| A builder is registered, but the op does not pass its tensors yet | `GemmOp` | error — **TileOPs is what needs changing** |
| No builder registered | everything else | error |

The second state follows from where TileOPs is in its migration: the place an op fetches
its kernel has to pass the tensors that kernel will be called with, or TileOPs cannot
compute the memo key for the external path.

```python
# inside TileOPs, in the op's own forward
self.get_or_build_kernel("gemm_kernel", (a, b), key=..., build=...)
#                                       ^^^^^^ this argument
```

**Only `RMSNormFwdOp` passes it today**; the other 84 kernel-fetch sites do not. The
change is mechanical and will be made op by op.

`build_gemm` in `src/tileops_cpu/pending.py` is exactly this case: correct, registered,
and currently unreachable. Writing the builder before TileOPs supplies that argument is
the normal order of work — the day `GemmOp` is wired up, this backend serves it unchanged.

### Platform predicates

Even for an op that reaches the external path, TileOPs' `forward` can still hold code
bound to specific hardware. `GemmOp` does:

```
tileops/ops/gemm.py:102       _get_kernel
tileops/kernels/call_spec.py  CallSpec.__post_init__
tileops/utils/utils.py:39     get_sm_version  ->  torch.cuda.current_device()
```

The selected target is a CPU one, and this code still queries the CUDA SM version. On a
machine with no CUDA driver, the call fails before it reaches `build_kernel`.

Around 94 such predicates exist in TileOPs. Clearing them is a precondition for the first
real heterogeneous backend, and will proceed family by family. Until then a backend author
meets them on their own hardware. File a TileOPs issue with the traceback when that
happens — TileOPs is what needs changing.

The example marks two tests `requires_cuda_runtime` for this reason; they skip
automatically on a machine with no GPU. What they check is TileOPs' platform assumptions,
not the backend.

## Error messages

All of these are measured output.

**A builder is registered, but the op does not pass its tensors yet:**

```
OpNotAvailableError: target 'torch_cpu' serves GemmOp, but its 'gemm_kernel' call site
does not hand over the tensors a builder is described with; that op is not wired to
external targets yet
```

A gap on the TileOPs side, not a backend problem. Wait for `inputs=` to be added, or file
an issue asking for that op to be prioritised.

**No builder registered for the op:**

```
OpNotAvailableError: target 'torch_cpu' registers no kernel builder for SoftmaxFwdOp;
targets that do: []. There is no fall back to the in-tree implementation: those kernels
do not run on this target's devices.
```

Write and register a builder for that op.

**An unregistered target was named:**

```
UnknownTargetError: no backend registered target 'nope'; known targets: ['torch_cpu']
```

The package did not install, or the target name is misspelled. Use
`tileops.backend.registered_targets()` to see what actually registered.

**`target=BUILTIN` forces the in-tree implementation:**

```
ValueError: RMSNormKernel is a CUDA kernel; got x on cpu and weight on cpu.
Another target's backend serves other devices.
```

`BUILTIN` bypasses backends explicitly. The in-tree implementation cannot run on CPU
tensors, which is precisely what the no-fall-back rule avoids.

**The backend package fails to import:** TileOPs skips it, warns, and collects the reason
into `load_failures()`. One broken plugin does not make TileOPs unimportable. If
registration raises part way through, everything that backend registered in that pass is
**rolled back** — no half-implemented target is left in the registry.

```python
from tileops.backend import load_failures
print(load_failures())
```

## Caller API

A backend author does not call these, but they help while debugging:

```python
from tileops.backend import (
    BUILTIN, registered_targets, set_default_target, default_target, load_failures,
)

registered_targets()                 # ['torch_cpu']
registered_targets("RMSNormFwdOp")   # ['torch_cpu']
set_default_target("torch_cpu")      # process default, ahead of device detection
set_default_target(BUILTIN)          # turn substitution off globally
```

Target selection order: the `target=` constructor argument → the process default → device
detection. There is no such thing as a "default target"; the default state is no
substitution.

## Running the tests

Needs an environment with `tileops` installed.

```bash
pip install -e .          # add --no-deps when tileops is already installed
python -m pytest -q       # with a GPU: 22 passed; without: 20 passed, 2 skipped
```

The same holds inside the TileOPs dev image, again without modifying TileOPs:

```bash
docker run --rm --gpus all -v "$PWD/..":/work -w /work \
  ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-afcebed1-dev \
  bash -lc 'pip install -e /work/TileOPs --no-deps -q &&
            pip install -e /work/tileops-backend-example --no-deps -q &&
            cd /work/tileops-backend-example && python -m pytest -q'
```

`tileops` is deliberately absent from the example's dependencies. The package extends an
installation that already exists, and a version floor here would resolve a release that
predates `tileops.backend`; the resulting `ImportError` is collected into
`load_failures()` and reads as "this backend is broken" rather than "TileOPs is too old".

## Phase limits

The decode path is captured by a CUDA graph, so each phase is bounded separately:

| Phase | May | May not |
| --- | --- | --- |
| memo lookup | a dict lookup | everything else |
| `detect` | one predicate | any import, any lock |
| kernel construction | select an implementation, compile, allocate, re-import, build handles | tuning that depends on real tensors |
| kernel call | launch a compiled kernel; allocate outputs through the torch allocator | compile, lazy init, build handles, host synchronisation |

**A module-level import must not trigger compilation.** TileOPs imports the backend module
while constructing the first Op; compilation belongs in `build_kernel`.

A kernel call has two further stream rules:

- **Launch on the current stream** (under CUDA, `torch.cuda.current_stream(device)`); never
  fall through to the default stream. Backends with their own launcher break this most easily.
- **Internal allocations must outlive asynchronous execution.** Where only a raw pointer is
  passed to a launch, the object has to stay alive until that stream has finished. The
  protocol provides no workspace; this safety is the backend's.

The caller warms up before capture — at least one non-captured call at the same shape —
because kernel construction is allowed to compile. During capture only one path is allowed:
memo hit, direct call.

## Out of scope

| Not supported | Reason |
| --- | --- |
| Two backends on one target | A target is one set of kernels from one provider. Registering the same `(op, target)` twice is an error — it means two packages both claim to be it |
| Replacing a composite op wholesale | A composite op's computation lives in the sub-ops it constructs; substitution happens at that level |
| A backend changing input shapes, or restoring outputs on the caller's behalf | That is what the op layer provides to every target; changing it means changing it for all of them |
| One call spanning several devices | CPU scalars travel as params, not tensor inputs; all inputs are on one device |
| A caller-provided workspace or explicit stream | What a backend needs is the current stream, and torch's stream is an implicit current value |
| autograd integration | This path serves inference. Forward and backward are separate ops |
| Substituting a different target | A named target without an implementation is an error; another target is not used instead |

## As a template

1. Copy the repository, rename `tileops_cpu` to `tileops_<hardware>`, and the target name with it.
2. Change `_detect` to claim the corresponding device type.
3. Replace `kernels.py` with real kernels — compile on construction, launch on `__call__`.
4. Pick the first op to take over and write its `build_kernel` against the op's manifest signature.
5. The four files under `tests/` carry over as they are; substitute the op and target names.
6. Add a `build_kernel` per op from there. **Every op the target model uses has to be covered** — a missing one is an error.
