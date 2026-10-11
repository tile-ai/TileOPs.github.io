# Adding a hardware backend

TileLang is a multi-backend DSL: each kind of hardware has its own set of kernels,
distributed as its own Python package. TileOPs therefore defines a protocol under which a
package outside the repository takes over an op's kernel in place of the in-tree
implementation, without any change to TileOPs.

This page describes how to bring in a new class of hardware, so that the ops on those
devices run that hardware's own kernels.

**A backend supplies one thing: a callable that computes this call.** The op layer does
everything else.

This page covers the target, the mechanism that takes over a whole op. The first half
describes the backend author's work, in the order it is done:

1. the four things to write;
1. the four functions of the protocol;
1. how one call reaches those functions;
1. a backend that installs and runs as it stands;
1. how to turn the template into a backend for real hardware;
1. the four rules for writing a kernel;
1. what each phase may do;
1. the state each op is in after install, and the cause behind each error message.

The second half explains why the protocol is designed this way:

- the two layers of selection;
- the op layer's contract;
- when a kernel is rebuilt;
- the interfaces available to a caller;
- what the protocol deliberately does not support.

## Two ways to extend dispatch {#two-ways}

A package outside TileOPs picks one of two mechanisms, by how much of an op it takes
over:

1. `register_kernel_type`: adds an implementation to a kernel interface, which takes
   part in selection alongside the in-tree implementations;
1. target: takes over every call of the op.

In the first, the kernel class follows the kernel interface, the same contract the
in-tree implementations follow; how to write one is in
[How a backend joins TileOPs](user-guide/dispatch/backends.md). A target follows the op's
manifest signature instead, and the backend writes a `build_kernel` for it. The rest of
this page covers the target only.

## Four things to write

| # | What to do |
| --- | --- |
| 1 | Declare an entry point in `pyproject.toml` that points at the backend module |
| 2 | Choose a target name and write a `detect`, declaring which class of devices these kernels are for |
| 3 | Pick the first op to take over, and write its `build_kernel` to the op's manifest signature |
| 4 | Call `register_detector` and `register_kernel_builder` at module top level |

Once these four are written, `pip install` makes the backend take effect. The following
sections give:

1. the signatures of the four functions;
1. how one call reaches them;
1. a [complete backend](#runnable) written to these four steps, which installs and runs as
   it stands.

After the first op, the backend adds a `build_kernel` per op. **Every op the target model
uses that builds kernels of its own must be covered.** A missing one is an error, with no
fall back to the in-tree implementation, because the in-tree kernels cannot launch on this
target's devices. A composite op, which only runs sub-ops, needs no builder.

## The protocol: four functions

`tileops.backend` defines only the outward-facing interface, expressed with Python
structural typing (`typing.Protocol`). A backend subclasses no base class and implements
no abstract method; it writes plain functions with matching signatures and registers them.
The op layer checks a backend's return value structurally as well, with `callable()`.

A backend implements `detect` and `build_kernel`, and registers them by calling
`register_detector` and `register_kernel_builder`. The protocol also defines `TensorSpec`,
and the backend declares one entry point in `pyproject.toml`:

| # | Name | Written by | Called by, and when |
| --- | --- | --- | --- |
| 1 | `detect` | implemented by the backend | called once for each target while the op layer settles the target for a call |
| 2 | `build_kernel` | implemented by the backend | called by the op layer on a memo miss |
| 3 | `register_detector` | the backend calls it | once, when the backend module is imported |
| 4 | `register_kernel_builder` | the backend calls it | when the backend module is imported, once per op it takes over |
| — | `TensorSpec` | defined by the protocol | built by the op layer and passed into `build_kernel` |
| — | the entry point | declared by the backend in `pyproject.toml` | enumerated by TileOPs when the first op is constructed |

The signatures follow in that order, with the protocol's `TensorSpec` last.

### 1. `detect`

```python
def detect(device: torch.device) -> bool: ...
```

Implemented by the backend. It answers whether this kind of device is served by this set
of kernels. It looks at the device only, not at dtypes or shapes. For a device that is not
its own, it returns `False` and does not raise.

```python
# claiming a whole device type
def detect(device: torch.device) -> bool:
    return device.type == "acme"

# reading an environment variable or asking a vendor runtime also belongs here
def detect(device: torch.device) -> bool:
    if device.type != "privateuseone":
        return False
    return acme_runtime.is_present(device.index)
```

### 2. `build_kernel`

```python
def build_kernel(*inputs: "TensorSpec | None", **params) -> Callable[..., KernelResult]: ...
```

Implemented by the backend, one per `(op, target)`. Its signature is the op's manifest
signature: `inputs` correspond one-to-one to `signature.inputs` in declaration order, and
`params` are named after `signature.params`. An optional input that was not passed on
this call arrives as `None`.

```python
# GroupNormFwdOp's spec: weight and bias are optional, and arrive as None when absent
def build_group_norm(x, weight, bias, *, num_groups, eps):
    if weight is None:                                   # presence read off the slot
        return AcmeGroupNorm(num_groups, eps, x.dtype)
    return AcmeGroupNormAffine(num_groups, eps, x.dtype)
```

### 3. `register_detector`

```python
def register_detector(target: str, detect: Callable[[torch.device], bool]) -> None: ...
```

Called by the backend, once per target, when the backend module is imported. It registers
that target's device detection function.

### 4. `register_kernel_builder`

```python
def register_kernel_builder(op: str, target: str, build_kernel: BuildKernel) -> None: ...
```

Called by the backend, once per op it takes over. It registers the kernel builder for
`(op, target)`; registering the same pair twice is an error.

### `TensorSpec`

```python
class TensorSpec(NamedTuple):
    device: torch.device
    dtype: torch.dtype
    shape: tuple[int, ...]
```

Defined by the protocol, built by the op layer and passed into `build_kernel`. It describes
a tensor's properties and does not contain the tensor.

```python
# what a build_kernel argument looks like
TensorSpec(device=torch.device("acme:0"), dtype=torch.float16, shape=(4096, 4096))

# and the three things it can be read for
def build_gemm(a: TensorSpec, b: TensorSpec, *, trans_a, trans_b):
    m, k = a.shape                    # shapes: compile-time constants, picking tiles
    if a.dtype is not torch.float16:  # dtype: raise here when it is unsupported
        raise ValueError(f"acme gemm needs fp16, got {a.dtype}")
    ...
```

The return value has one structural requirement: **it must be callable**. It is called
as `(*tensors)` and returns a tensor, a tuple of tensors, or `None` for a pure in-place
write. The op layer checks it with `callable()`.

**The protocol passes only descriptions of tensors.** The protocol therefore needs no
separate rule that "a builder must not read tensor contents or keep a reference to a
tensor", a rule the op layer could not enforce. Such a rule would guard against two
things:

- **Reading data** would make the built kernel depend on data, while the memo table
  keys only on device and shape.
- **Keeping a reference** would keep a tensor alive as long as the cached kernel.

A `TensorSpec` carries neither data nor a tensor, so neither can happen.

The next section shows when each of the four functions is called during a real call.

## How one call reaches `build_kernel` {#from-op-layer}

The steps of one call, from user code to a backend's `build_kernel`:

```python
# ── the caller ───────────────────────────────────────────────────────
op = GemmFwdOp()                 # no target= in the constructor, so the inputs' device decides
                                 #   target="acme" skips detection and uses it directly;
                                 #   target=BUILTIN forces the in-tree kernels
a = torch.randn(4096, 4096, dtype=torch.float16, device="acme:0")
b = torch.randn(4096, 4096, dtype=torch.float16, device="acme:0")
d = op(a, b)                     # every input on one device: a.device == b.device

# ── op layer: settle the target ──────────────────────────────────────
# Every installed backend put a detect in the registry when it was imported, and the op
# layer passes a.device to each of them in turn and asks "is this device yours?":
#   acme's detect(device) → True       every other backend's → False
#   exactly one True   → target = "acme", and this instance keeps it from here on
#   none True          → the in-tree kernels run
#   two or more True   → AmbiguousTargetError, asking for an explicit target=

# ── op layer: run the checks generated from the manifest signature, then hand the whole op to the target ──
#   GemmFwdOp's own forward and kernel_for serve the in-tree path only, and do not run
#   the tensors go to the target in signature.inputs order, inputs it does not write made contiguous

# ── op layer: look up the external memo table — device, then input signature ──
#   ("acme:0", (float16, (4096, 4096)), (float16, (4096, 4096)))
#   the device first, then one (dtype, shape) per input
#   first call on this instance, so the table is empty → a miss, and it builds
#   a later call at the same device, dtypes and shapes hits and jumps to the last step

# ── backend: the op layer calls build_gemm, with TensorSpecs, not tensors ──
#   build_gemm(TensorSpec("acme:0", float16, (4096, 4096)),
#              TensorSpec("acme:0", float16, (4096, 4096)),
#              trans_a=False, trans_b=True)      # params by their manifest names
#   → returns something callable

# ── op layer: store it, then launch ─────────────────────────────────
#   kernel(a, b)                 # d = a @ b.T, computed by acme's kernel
```

A backend writes one of these steps, `build_gemm`, and registers it:

```python
def build_gemm(a: TensorSpec, b: TensorSpec, *, trans_a, trans_b):
    m = a.shape[1] if trans_a else a.shape[0]
    if m == 1:                                  # the backend decides the case from the specs
        return AcmeGemv(a, b, trans_a, trans_b)
    return AcmeGemm(a, b, trans_a, trans_b)


register_kernel_builder(op="GemmFwdOp", target="acme", build_kernel=build_gemm)
```

The op layer calls `build_gemm`; the backend never calls it itself. Importing the backend
module only records it in the registry. It is called when this target serves an op call
and the external memo table misses, once per device and input signature. The op layer
stores the callable it returns in the memo table and launches it.

This call path implies four things:

- **`kernel_for` and the implementations' `entry_for` serve the in-tree path only.** They
  decide which in-tree kernel is fetched, what it is looked up by and how it is built.
  Once the op settles on a target, that target serves the whole op, and none of them runs.
- **Tensors arrive positionally, params by name.** `build_kernel(*inputs, **params)`: the
  positional arguments are `TensorSpec`s (`None` for an optional input the call omitted),
  and the keyword arguments are the manifest's `params` names with the values settled for
  this call.
- **One builder per `(op, target)`.** The backend is not told which kernel the in-tree
  path would run (GEMM declares three in `kernel_types`); `build_kernel` decides from the
  `TensorSpec`s which kernel to return.
- **The backend needs no memoisation of its own.** For the same device and input signature
  the op layer does not call `build_kernel` again. For a finer split, or fewer rebuilds,
  the backend adds a cache inside `build_kernel`. An op written only for external backends
  declares neither `kernel_types` nor `interfaces`; a call on it with no target claiming
  the device raises `OpNotAvailableError`.

## Writing a backend that runs {#runnable}

[`tileops-backend-example`](https://github.com/lcy-seso/tileops-backend-example) is a
complete backend written to the four steps, and it can be copied as a starting point. It
implements its kernels in pure PyTorch and claims CPU, so it installs, runs and tests on
any machine. Apart from kernels that use no dedicated hardware, every part of it is what a
backend for dedicated hardware writes: the entry point, registration, the `build_kernel`
signature, the memoisation rule and the error messages.

The difference before and after installing it:

```console
$ python -c "import torch; from tileops.norm import RMSNormFwdOp; \
             RMSNormFwdOp(normalized_shape=(64,))(torch.randn(4,64,dtype=torch.float16), \
                                                  torch.randn(64,dtype=torch.float16))"
OpNotAvailableError: RMSNormFwdOp's in-tree kernels do not run on cpu; known targets for this op: []

$ pip install -e .

$ python -c "...the same code..."
# returns normally, bit-identical to torch.nn.functional.rms_norm
```

The following describes its contents, one step at a time.

**Step 1, three lines of `pyproject.toml`.** The entry-point group is always
`tileops.backends`, and the value is the backend's module:

```toml
[project.entry-points."tileops.backends"]
torch_cpu = "tileops_cpu"
```

After `pip install`, no initialisation is needed. TileOPs enumerates this group while
constructing its first op and imports the module named there; the registration calls at
module top level fill the registry. The backend inherits no base class and implements no
interface.

**Step 2, a target name and a `detect`.** Both are in `target.py`. `detect` is a single
line and claims every CPU device:

```python
TARGET = "torch_cpu"


def detect(device: torch.device) -> bool:
    return device.type == "cpu"
```

The two names mean different things: `TARGET = "torch_cpu"` names this set of kernels and is
the backend author's to choose, while `device.type == "cpu"` is the device type it claims,
defined by torch.

**Step 3, a `build_kernel` written to the manifest signature.** `RMSNormFwdOp`'s spec
declares two inputs, `x` and an optional `weight`, and two params, `normalized_shape` and
`eps`; the function's parameters follow that declaration. It is in `ops/rms_norm.py`,
together with the kernel class `CpuRMSNorm`:

```python
def build_rms_norm(x: TensorSpec, weight: TensorSpec | None, *, normalized_shape, eps):
    if eps is None:                     # a null manifest default: the ref API's meaning
        eps = torch.finfo(torch.float32).eps
    return CpuRMSNorm(normalized_shape, eps, x.dtype)
```

**Step 4, the registrations at module top level.** `BUILDERS` in `ops/__init__.py` lists
every op the target takes over, with each key spelled as in the manifest. The package's
`__init__.py` registers one detector, then one builder per entry:

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

## The template project: layout, tests, and retargeting it

### Repository layout

Each file in the example covers one part of the backend author's work:

| File | Contents |
| --- | --- |
| `pyproject.toml` | the entry point declaration, which is the whole install mechanism |
| `src/tileops_cpu/__init__.py` | all registration code |
| `src/tileops_cpu/target.py` | the target name and `detect` |
| `src/tileops_cpu/ops/__init__.py` | `BUILDERS`, the table of every op the target takes over. Its keys are spelled as in the manifest; a builder under a misspelt key is never called |
| `src/tileops_cpu/ops/rms_norm.py`, `ops/gemm.py` | one module per op, holding the kernel and the builder that constructs it. A real backend compiles in the kernel's constructor |
| `tests/test_takeover.py` | numerics, validation, normalisation, outputs |
| `tests/test_discovery.py` | entry point and registration |
| `tests/test_errors.py` | the three error paths: an unregistered op raises rather than falling back, an unknown target raises, and a failed call binds the op to no target |
| `tests/test_memoization.py` | when `build_kernel` is called again |

`CpuRMSNorm` does not receive the row count when it is constructed. This is the rule
"the constructor takes compile-time parameters only" in practice.

### Running the tests

The example's tests need an environment with `tileops` installed:

```bash
pip install -e .          # add --no-deps when tileops is already installed
python -m pytest -q       # 24 passed, both with two H200s visible and with CUDA_VISIBLE_DEVICES=""
```

The tests also run inside the TileOPs dev image, again without modifying TileOPs:

```bash
docker run --rm --gpus all -v "$PWD/..":/work -w /work \
  ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-afcebed1-dev \
  bash -lc 'pip install -e /work/TileOPs --no-deps -q &&
            pip install -e /work/tileops-backend-example --no-deps -q &&
            cd /work/tileops-backend-example && python -m pytest -q'
```

`tileops` is deliberately absent from the example's dependencies. The package extends an
installation that already exists, and a version floor here would resolve to a release
that predates `tileops.backend`. The resulting `ImportError` is collected into
`load_failures()` and appears as "this backend is unusable", while the real cause is that
TileOPs is too old.

### Turning it into a backend for real hardware

1. Copy the repository, rename `tileops_cpu` to `tileops_<hardware>`, and the target name with it.
2. Change `detect` in `target.py` to claim the corresponding device type.
3. Replace the kernels in the `ops/` modules with real ones, which compile on construction and launch on `__call__`.
4. Pick the first op to take over, write its `build_kernel` against the op's manifest signature, and add it to `BUILDERS`.
5. The four files under [`tests/`](https://github.com/lcy-seso/tileops-backend-example/tree/main/tests) carry over largely as they are; substitute the op and target names.
6. Add a `build_kernel` per op from there, until every op the target model uses is covered.

## Writing a kernel

### The signature comes from the manifest

**Writing a kernel needs the manifest, not the TileOPs source.** A builder's signature
is the op's manifest signature. For example, `RMSNormFwdOp` in
[`src/tileops/manifest/spec/norm.yaml`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/manifest/spec/norm.yaml)
declares:

```yaml
signature:
  forall: {B: Shape, T: "DType[float16 | bfloat16]"}
  params:                       # passed as keyword arguments under these names
    normalized_shape: {type: "list[int] | tuple[int, ...]"}
    eps: {type: "float | None", default: null}
  inputs:                       # declaration order is call order
    x: {dtype: T, shape: "[*B, *normalized_shape]"}
    weight: {dtype: T, shape: "[*normalized_shape]", optional: true}
```

The corresponding builder signature:

```python
def build_rms_norm(x: TensorSpec, weight: TensorSpec | None, *, normalized_shape, eps):
```

The signature follows two conventions:

- **Parameters arrive as the op instance holds them.** An `eps` not given at construction
  arrives as the manifest default, `None`, with the meaning it has in the reference API,
  and the builder handles it with that meaning. An omitted optional `weight` arrives as
  `None`.
- **The return value follows `signature.outputs`:** a tensor for a single output, a tuple
  in declaration order for several outputs, and `None` for a pure in-place write.

### The constructor takes compile-time parameters only

Values compiled into generated code (tile sizes, dimensions treated as constants, dtypes)
go in the constructor; all other values belong to `__call__`.

On the decode path this rule is a hard requirement: `seq_len` grows step by step and
batch changes with the running set, so putting them in the constructor recompiles the
kernel at every step.

### Shapes are the manifest's

The op layer changes no shapes. A kernel receives the shapes the manifest declares, and
arranges any layout it needs inside its own call wrapper.

Where the code and the manifest disagree, the manifest governs. The manifest defines the
output dtype, the shape rules and the parameter types, and a kernel does not rewrite them.

### What a kernel's error has to say

A kernel that cannot serve a call raises instead of degrading. Its error states two
things:

- **Which item is unmet:** dtype, shape, arch, no implementation available, or
  compilation failed.
- **The value it actually received.**

"Unsupported" on its own is not a diagnosis.

## What each phase may do {#phase-limits}

The decode path is captured by a CUDA graph, so each phase has its own limits:

| Phase | May | May not |
| --- | --- | --- |
| Memo lookup (its key and rebuild rules are in [When a kernel is rebuilt](#memo)) | one dict lookup | anything else |
| `detect` | one predicate | any import, any lock |
| Building a kernel | select an implementation, compile, allocate, re-import, build handles | tuning that depends on real tensors |
| Calling a kernel | launch a compiled kernel, allocate outputs through the torch allocator | compile, lazy init, build handles, host-side synchronisation |

**A module-level import must not trigger compilation.** TileOPs imports the backend
module while constructing the first op; compilation belongs in `build_kernel`.

A kernel call also follows two stream rules:

- **It launches on the current stream.** Under CUDA, the current stream is
  `torch.cuda.current_stream(device)`; a kernel never falls through to the default stream.
  Backends with their own launcher break this rule most easily.
- **Internal allocations must outlive asynchronous execution.** When only a raw pointer is
  passed to a launch, the object must stay alive until that stream has finished. The
  protocol provides no workspace; the backend is responsible for this.

Because building a kernel may compile, the caller warms up before capture, with at least
one non-captured call at the same shape. During capture only one path is allowed: a memo
hit, then the call.

## After install: three states {#three-states}

Once `detect` claims a class of devices, **every** op on those devices is served by that
target. A missing op is an error, with no fall back to the in-tree implementation. The
one exception is a composite op, which builds no kernel of its own.

The op layer does not fall back because selecting a target means the device belongs to
other hardware, where the in-tree kernels cannot launch at all. Falling back would replace
a clear "this target does not implement this op" error with an obscure launch failure.

After install, each op is therefore in one of three states:

| State | Result |
| --- | --- |
| The target registered a `build_kernel` for the op | it runs, the whole op on the target |
| It did not, and the op builds kernels of its own | an error naming the target and the op, with no fall back to the in-tree implementation |
| It did not, and the op is a composite | the op runs its composition, and each sub-op settles on a target itself |

Covering every op the target model uses is therefore the backend's work. The op side is
settled by design: the op layer computes the memo key of the external path from the call's
own inputs (see [How one call reaches `build_kernel`](#from-op-layer)).

### No hardware queried before the target is settled

Until a target is settled, the op layer queries nothing bound to specific hardware, such
as a CUDA SM version. With such a query, on a machine without that driver the call would
fail before it reaches `build_kernel`, for a reason unrelated to the backend.

If such a failure occurs on your hardware, the traceback stops inside TileOPs, not in the
backend's `build_kernel`. That is a regression on the TileOPs side: file an issue with the
traceback.

Every test in the example passes where no GPU is visible, and none is skipped; that run
checks this premise.

## Error messages and what to do

The three messages below are measured output. Each has one cause and one fix.

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
OpNotAvailableError: RMSNormFwdOp's in-tree kernels do not run on cpu; known targets for this op: ['torch_cpu']
```

`BUILTIN` bypasses all backends explicitly. The in-tree implementation cannot run on CPU
tensors; this error shows the outcome that the no-fall-back rule avoids.

**When a backend package fails to import**, TileOPs skips it, issues a warning, and
collects the reason into `load_failures()`. One broken plugin does not make TileOPs fail
to import. If registration raises part way through, everything that backend registered in
that pass is **rolled back**, and no partly registered target is left in the registry.

```python
from tileops.backend import load_failures
print(load_failures())
```

## Why selection has two layers

| Name | Definition | Where it sits in dispatch |
| --- | --- | --- |
| **target** | The name of one set of kernels. A backend release brings a set of kernels and gives it a name, e.g. `"acme"` | The first layer: pick a target, and this call's kernel comes from its set |
| **`detect`** | A function a backend writes, one per target | How the first layer picks: it receives a `torch.device` and answers whether such a device is what its kernels are for; `False` if not |
| **`build_kernel`** | A function a backend writes per op, one per `(op, target)` | The second layer: it receives a description of this call — each input's device, dtype and shape, plus the op's parameters — and picks, builds and returns a kernel from its own set |

**Selection has two layers: TileOPs picks the target, and the target picks the kernel
from its own set.** The second layer happens inside `build_kernel`, without the protocol:
this path has no kernel-level concept, no capability negotiation and no candidate
filtering. The candidate filtering TileOPs does run (availability, applicability,
precedence) belongs to the in-tree path and to the smaller mechanism, `register_kernel_type` (see
[Two ways to extend dispatch](#two-ways)); a target bypasses it.

`detect` answers only which devices belong to the backend, and nothing finer.
**Whether this call is supported (dtype, shape, parameter combination) is answered by
`build_kernel`**, the only place that sees the full input description and the
parameters; it raises there when it cannot serve the call. `detect` cannot make these
judgements, because it receives only a `torch.device`.

TileOPs does not parse `torch.device`; it passes it unchanged to `detect`. Device types
and targets are not in one-to-one correspondence:

- one device type can carry several sets of kernels from different vendors;
- some hardware arrives through `privateuseone`, whose string carries no vendor
  information;
- some backends must read an environment variable or call a vendor runtime to decide.

## The op layer's contract

The seven items below are the op layer's contract to every target. The op layer
implements them, and a backend reuses them instead of writing its own. They are listed in
the order a backend author meets them:

| # | The op layer supplies | What it means for a backend |
| --- | --- | --- |
| 1 | The public torch-side API and the meaning of each parameter | How the op is called, the parameter names and their semantics are settled; a backend neither defines nor changes them |
| 2 | Manifest validation | A call whose dtype or shape does not conform is rejected at the op layer and never reaches the backend |
| 3 | Parameters by name | Parameters arrive under the manifest's `params` names, with the values the op instance holds. For a parameter the manifest defaults to null, that is the value the op settled on, either `None` or a number |
| 4 | Input contiguity | Every input the call does not write arrives contiguous; an input it writes arrives as the caller passed it, unless the manifest declares it `contiguous: true` |
| 5 | Memoisation and reuse of kernels | The builder is called once per specialization: a later call with the same device and input signature reuses the previous return value. A builder may therefore compile, and the op layer guarantees it is not called again |
| 6 | The `torch.compile` and CUDA-graph boundary | The op layer wraps a call as an opaque operator and registers a fake alongside, so the compiler can infer the output's shape and dtype without executing. **A backend's kernels do nothing for compilation**; see [Bringing an op into torch.compile](torch-compile.md) |
| 7 | Roofline, profiling and numerical tests | The op layer's existing tests run once with the backend's kernel and compare against the manifest's `ref_api`; performance reports are produced as usual |

None of the seven depends on hardware, and every target gets them identically. A
third-party backend neither bypasses any of them nor substitutes its own.

The in-tree kernels ([`src/tileops/kernels/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/kernels))
are the **default implementation**: they have no target name and are not in the registry.

**The in-tree implementations run by default.** With no backend installed, no `target=`
named and no process default set, calls run the in-tree implementation. An installed
backend serves an op only when its `detect` claims the device, or when it is named by
`target=` or `set_default_target`.

## When a kernel is rebuilt {#memo}

TileOPs memoises a builder's return value by **device plus input signature**:

> the device this call's tensors are on, plus `(dtype, shape)` taken per input in
> `signature.inputs` order; an optional input not passed on this call is recorded
> as `None`.

**Two calls with the same device and input signature get the same kernel**, with no
further call to `build_kernel`. A second card under the same target builds again, because
an artefact compiled for one device need not launch on another. Params are not part of
the key, because they are fixed for an op instance.

How the op layer looks up that table, and what it does on a miss, is in
[How one call reaches `build_kernel`](#from-op-layer).

Two consequences:

- **An entry may not last.** When a call fails, the op revokes its target decision and
  drops its memo table. A backend must not assume the callable it returned stays alive;
  the callable holds its own references to the resources it depends on.
- **A finer or a coarser grain is handled on the backend side.** Finer distinctions are
  made inside the backend; to rebuild less often, the backend adds a cache inside
  `build_kernel`.

## What a caller can reach for

These interfaces are for callers. A backend author does not need to call them, but they
help while debugging:

```python
from tileops.backend import (
    BUILTIN, registered_targets, set_default_target, default_target, load_failures,
)

registered_targets()                 # ['torch_cpu']
registered_targets("RMSNormFwdOp")   # ['torch_cpu']
set_default_target("torch_cpu")      # process default, ahead of device detection
set_default_target(BUILTIN)          # turn substitution off globally
```

The target is selected in this order:

1. the `target=` constructor argument;
1. the process default;
1. device detection.

`BUILTIN` forces the in-tree implementation. A named target that is not registered, or
does not implement the op, is an error; another target is not used instead.

## What the protocol does not support

The following cases are outside the protocol, each for the reason given:

| Not supported | Reason |
| --- | --- |
| Two backends on one target | A target is one set of kernels from one provider. Registering the same `(op, target)` twice is an error, because it means two packages both claim to serve it |
| Falling back across targets | A named target without an implementation is an error; another target is not used instead |
| A backend changing input shapes, or restoring outputs on the caller's behalf | That is what the op layer provides to every target; changing it means changing it for all of them |
| One call spanning several devices | All inputs are on one device, except the tensors the manifest declares `device: cpu` |
| A caller-provided workspace or explicit stream | What a backend needs is the current stream, and torch's stream is an implicit current value |
| autograd integration | This path serves inference; forward and backward are separate ops |
