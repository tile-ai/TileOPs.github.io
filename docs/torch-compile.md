# Bringing an op into torch.compile

A TileOPs op brought into `torch.compile` becomes one node in the user's compiled
graph, and that node does not change with the backend serving it.

An op is brought in through a compile boundary at the op layer. Dynamo traces
everything outside the boundary, and everything inside it is invisible to the compiler.
The base class generates this boundary when the class is defined, for every op whose
manifest entry has a call-time tensor input and no composition; the op writes no code
for the boundary.

The body covers the work of bringing an op in:

1. checking whether an op is already in;
1. compiling code that calls it;
1. the five calling conventions;
1. how the boundary is generated, and the code the op writes.

The appendix explains why the boundary can only be drawn this way: how dynamo works,
where it and the op layer disagree, why the boundary sits at the op layer, and what the
boundary costs and does not provide.

## Calling an op that is already in

### Checking whether an op is in {#supported}

Read the class attribute `compile_op_names`. A non-empty value means the class has a
compile boundary, so the boundary is at the op layer and `fullgraph=True` works. An
empty tuple means the class has no compile boundary.

```python
>>> from tileops.norm import RMSNormFwdOp
>>> RMSNormFwdOp.compile_op_names
('tileops::norm_rms_norm_fwd',)
```

An op class has a compile boundary exactly when its manifest entry meets both of:

1. it has a call-time tensor input;
1. it has no composition, that is, the op is not a composite.

Every implemented op that meets both must pass a cold
`torch.compile(op, fullgraph=True)`. The validator checks each implemented entry's
`compile_op_names` against the two conditions, and a test further requires each such
op to register a cold compile test in
[`tests/compile_contract.py`](https://github.com/tile-ai/TileOPs/blob/main/tests/compile_contract.py).

Two kinds of op have no compile boundary: ops with no call-time tensor input
(`AlibiFwdOp`, `SinusoidalFwdOp`) and composites. When a composite is traced, dynamo
traces its `forward`, and each sub-op it calls that has a compile boundary becomes a
node in the graph.

### Compiling code that calls it

Construct the op instance and pass the function that calls it to `torch.compile`. No
other step is needed:

```python
import torch
from tileops.norm import RMSNormFwdOp

op = RMSNormFwdOp(normalized_shape=(4096,))     # construct once, reuse

@torch.compile(fullgraph=True)
def block(x, weight):
    return op(x, weight)

x = torch.randn(2048, 4096, device="cuda", dtype=torch.float16)
w = torch.randn(4096, device="cuda", dtype=torch.float16)
block(x, w)
```

Running with `TORCH_LOGS=graph_code` prints the captured graph. It has one node,
`tileops::norm_rms_norm_fwd`, and the calls inside the kernel do not appear in it.

### The five calling conventions

Each convention follows from a mechanism at the boundary. Breaking any of them makes
the compiled path behave differently from the eager one.

- **Construct the op instance once and reuse it.** The instance key is a compile-time
  constant and each instance has its own compiled graph, so constructing an instance
  inside a loop recompiles on every iteration.
- **Strides are not passed through.** A non-contiguous input the op does not write is
  made contiguous inside the node, and an output the op allocates is always contiguous.
  When later work needs another layout, convert outside the op. An output that is a
  written input (`alias`) or a caller's `out` keeps that tensor's storage.
- **Meta tensors cannot be used for warm-up.** For an op with a compile boundary, a call
  with meta or fake tensors returns at the fake and never reaches kernel construction.
- **Warm up before a CUDA graph capture.** Call the op at least once with real tensors
  at the same shape: building a kernel may compile, while a capture allows only a cache
  hit followed by the call. What each phase allows is in
  [what each phase may do](backends.md#phase-limits).
- **A second card may need its own build.** For a call a target serves, the device is
  part of the kernel's cache key, so the same instance builds the kernel again on a
  second card. The cache key of an in-tree kernel is the build identity the selected
  implementation's `entry_for` returns, which includes the device only when the build
  depends on it. A `target=` given to the constructor also takes effect on the first
  compiled call, and a failed build pins the op to no target.

### The three guarantees once an op is in

With the boundary at the op layer, a caller can rely on three things:

- **The graph does not change with the target.** The same code compiles to the same
  graph on another backend or another card, so the compiled artefact does not depend on
  the backend.
- **`fullgraph=True` works** for an op with a compile boundary; see
  [Checking whether an op is in](#supported).
- **Output shape, dtype and stride come from the manifest.** They do not depend on how a
  kernel tiles or pads internally. An output the op allocates is always contiguous.

## How the boundary is generated: `RMSNormFwdOp`

This section gives the code an op with a compile boundary writes, how the boundary and
its fake are generated, and why the target is resolved inside the node. The
tracing, graph breaks and guards it refers to are described in
[How dynamo works](#dynamo).

`RMSNormFwdOp`'s skeleton, with docstrings elided; the
full file is
[`src/tileops/ops/norm/rms_norm.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/ops/norm/rms_norm.py):

```python
class RMSNormFwdOp(Op):
    # the operators, their fakes, _call_boundary and compile_op_names are generated
    # from the manifest entry
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "rms_norm": RMSNormKernel,
        "rms_norm_streaming": RMSNormStreamingKernel,
        "rms_norm_on_chip": RMSNormOnChipKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"rms_norm": RMSNormFwdInterface}

    def __init__(self, normalized_shape, eps=None, *, target=None):
        self.normalized_shape = normalized_shape
        self.eps = eps
        super().__init__(target=target)

    def forward(self, x, weight=None):
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

The op writes only its constructor and `forward`, its computation. `Op.__call__` calls
the generated `_call_boundary`, which takes `forward`'s parameters and calls the
operator of the call's effect branch; inside the node, the operator runs the call with
`forward` as its body. The operators and their fakes are generated from the manifest
entry, one operator per effect branch:

- its tensor arguments are `signature.inputs`, in order;
- its return value is given by `signature.outputs`;
- the arguments it writes are exactly the inputs marked `mutated`;
- each output's shape and dtype come from the signature.

The operator's name is `tileops::<family>_<snake(class)>`, with the family written only
once when the class name already starts with it; here the name is
`tileops::norm_rms_norm_fwd`. When a branch writes an input, fills a `buffer: out` or
omits an output, its operator's name gains `_writes_<input>`, `_out` and
`_without_<output>` respectively, in that order. No op names its own operator, so
`compile_op_names` cannot disagree with the registered names.

The layers one call passes through, and where the boundary falls:

<figure class="callpath" markdown="0">
  <div class="cp-step cp-traced"><code>Op.__call__</code><span>calls the generated <code>_call_boundary</code>, resolves no target</span></div>
  <div class="cp-step cp-traced"><code>_call_boundary</code><span>takes <code>forward</code>'s parameters, calls the opaque operator</span></div>
  <div class="cp-boundary"><span>compile boundary</span></div>
  <div class="cp-step cp-opaque"><code>the generated operator</code><span>recovers the instance, runs the generated checks, resolves the target, undoes the resolution on failure</span></div>
  <div class="cp-step cp-opaque"><code>forward</code><span>makes the inputs contiguous, fetches the kernel, launches it</span></div>
  <figcaption>The two violet layers are inside dynamo's trace, and <code>_call_boundary</code>'s call of the operator is the last thing dynamo traces. Below the boundary the opaque operator runs, invisible to the compiler.</figcaption>
</figure>

Three parts of the generated code are fixed.

**First, the instance is recovered through a string key, not passed as an object.**
The schema's types are a fixed set, such as `Tensor`, `int`, `float`, `bool` and
`str`, with no "arbitrary Python object". What the operator body needs (the resolved
target and the cache table of built kernels) is stored on the instance and
cannot be split into schema arguments. Two details of the key are also fixed:

- **The key is a string, not an integer.** A string is a constant during tracing, while
  an integer is generalised to a `SymInt`.
- **A key is never reused.** Because the key is a constant, inductor bakes the shape the
  fake gave into the compiled artefact, and an op that reused a key would inherit the
  previous instance's shape.

**Second, the fake builds its output with `torch.empty` from the shape and dtype the
signature check infers, not with `torch.empty_like(x)`.** The tensor the fake returns
must match real execution in shape, dtype and stride. A mismatch either fails during
tracing or, for a stride, makes downstream code read memory in the wrong layout and
produce wrong results silently. The operator body makes the inputs contiguous before the
kernel writes into a newly allocated output, so the real output is always contiguous.
`empty_like` copies the input's strides, so for a non-contiguous input the fake would
declare a layout real execution never produces.

**Third, the target is resolved inside the node, not in `Op.__call__`.** When traced
code runs `self.x = ...`, dynamo records the write as a pending side effect and applies
it only after the whole graph has run, while the opaque node runs before that. A
resolution written just outside the node therefore cannot be read inside it. Two things
follow, and both happen inside the node:

- A resolution made outside the node would make the first compiled call silently run
  the wrong implementation.
- Undoing a failed resolution is the job of the place that made it, because a compiled
  artefact does not keep the call site's `try/except`.

All three have the same cause: torch's compilation and declaration mechanisms work per
function, while what needs compiling is one call on an object.

## Appendix: why the boundary looks like this

### How dynamo works {#dynamo}

This section describes how dynamo decides what code may enter a compiled graph. The
conditions an op has to satisfy to be brought into `torch.compile` come from these
rules.

Dynamo is the front end of `torch.compile`, and works at CPython's frame evaluation
layer (PEP 523).

**Dynamo has exactly one entry point: `torch.compile`.** `torch.compile(fn)` returns a
wrapper, and tracing happens when the wrapper is called; `nn.Module.compile()` and the
decorator form are two other spellings of the same entry. A call that does not go
through it takes the ordinary Python path and has nothing to do with dynamo. This page
calls that path eager.

On the first call, dynamo takes over the frame, symbolically executes the bytecode
instruction by instruction, records the tensor operations as one FX graph, and leaves
what cannot enter the graph to run in Python. It also records a set of guards for the
graph: the premises this trace relied on, such as a tensor's dtype and rank. A later
call reuses the compiled artefact when every guard holds; when one fails, dynamo traces
the new case again.

Three terms have fixed meanings on this page:

| Term | Meaning |
| --- | --- |
| Graph | the FX graph dynamo captured; one trace produces one |
| Node | one operator call in the graph, with input edges and the output's shape and dtype |
| Traced | inside dynamo's symbolic execution. Tracing performs no real computation; it only records |

The graph then goes to a compiler backend (inductor and others) for fusion, memory
planning and code generation. The larger a graph is, the more neighbouring operators
can fuse, so every op in an operator library has to be able to appear as a node in a
user's graph.

Two of dynamo's rules decide how an op is brought in:

- **Dynamo inlines by default.** A called function is not itself a boundary, and its
  body is folded into the same trace. Keeping a stretch of Python code out of the trace
  requires an explicit declaration.
- **Untraceable code is handled in one of two ways.** By default dynamo breaks the
  graph: that stretch falls back to Python, and one graph becomes several. Under
  `fullgraph=True` dynamo raises instead. Raising surfaces the problem during
  development, which is why an operator library uses `fullgraph=True` as its acceptance
  criterion.

### Where the op layer and dynamo disagree

Applying those rules to a TileOPs op shows the obstacle. Dynamo compiles frames, that
is, functions, while a TileOPs op is an object. One call does four things, and only the
last belongs in the graph:

| What a call does | Should dynamo capture it |
| --- | --- |
| Validate dtypes and shapes, make the inputs contiguous | No |
| Decide which target serves this call | No |
| Fetch or build the kernel | No — capturing this far fails |
| Launch the kernel and produce the output | Yes, as a node in the graph |

The table needs three qualifications.

**Work that should not be captured still runs.** All four things happen on every call;
the only difference is whether they enter the graph.

**The distinction has to be annotated by hand**, because dynamo cannot draw it. Torch
provides two interfaces for this:

- `torch.library.custom_op` registers the call as an operator, so dynamo puts a single
  node in the graph and does not trace into the implementation;
- `register_fake` tells the compiler what the node outputs; it receives only the
  inputs' metadata and never touches real data.

**Without the annotation, dynamo traces into the code and fails.** If `RMSNormFwdOp`
had no compile boundary, it would compile in neither of its two states:

- An instance that has not built a kernel builds one during the call, and dynamo
  traces into the TileLang JIT inside the constructor.
- An instance that already has one skips construction, but still re-parses the
  TileLang program on every call, so dynamo traces into `@tilelang.jit` and stops at
  `inspect.signature`.

### Why the boundary falls at the op layer {#at-op-layer}

The boundary could sit at the op layer or lower, at the kernel layer. The difference
shows in the user's compiled graph.

The node's identity in that graph (its name, arguments, granularity, and the output
its fake declares) is the op the user sees. With the boundary at the kernel layer,
changing backend changes that node: the same op compiles to a different graph under a
different target, and the compiled artefact is tied to the backend. With the boundary
at the op layer, the node's identity is the op's, independent of the backend serving
it.

That position also decides how the fake is written. The op layer does not know how an
external kernel tiles or pads internally. The only shape rule that holds for every
target is the one in the manifest, so the fake derives its output from the manifest.

The node's interior is invisible to the compiler, but its contract with the outside is
complete:

- the schema gives the name and argument types;
- the fake gives the output's shape, dtype, device and stride;
- the alias annotations name exactly the inputs the node writes, of which
  `RMSNormFwdOp` has none.

Because the contract is complete, what the boundary keeps and what it gives up
separate cleanly:

- **Optimisation between nodes proceeds as usual.** This includes buffer assignment,
  lifetimes, reordering against neighbours the node does not depend on, and deleting the
  node when nothing consumes its output.
- **Optimisation inside the node is lost.** Neighbouring operators cannot fuse into
  the node, and its output must be written to memory.

For an operator library the trade is worth it: inside the node is a kernel TileLang has
already compiled, which inductor does not need to touch.

### What the boundary costs

Measured on an idle H200 at 2048×4096, fp16. Per-call figures are the minimum of three
runs of 2000 iterations × 9 rounds:

| | Boundary at the kernel layer | Boundary at the op layer |
| --- | --- | --- |
| Kernel time | 0.0119 ms | 0.0117 ms |
| Eager, per call | 42.5–45.2 µs | 38.2–42.0 µs |

The kernel itself is unaffected, because where the boundary sits has nothing to do
with how the kernel computes. The eager path is 3–5 µs faster because, with the boundary
moved up, a call crosses one operator boundary instead of two.

The cost on the graph side is described in
[Why the boundary falls at the op layer](#at-op-layer): fusion does not cross the node
boundary, and the node's output is always written to memory.

### What the boundary does not provide

| Not provided | Reason |
| --- | --- |
| Fusion across the node boundary | The node's interior is opaque to the compiler, so the elementwise work on either side stays outside |
| autograd through the node | This path serves inference; forward and backward are separate ops |
| Switching target within one compiled artefact | The target belongs to the op instance: another target means another instance, and another graph |
| Building a kernel from meta tensors | A call with meta tensors returns only shapes and dtypes |
