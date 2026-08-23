# Bringing an op into torch.compile

Bringing a TileOPs op into `torch.compile` makes it one node in the user's compiled
graph, and that node's shape does not change with the backend serving it.

It takes one thing: a compile boundary declared at the op layer, with everything outside
it traced by dynamo and everything inside it invisible to the compiler.

The body covers the work — checking whether an op is already in, compiling a call, the
five conventions, and the declaration an op that is not in yet needs. The appendix covers
why the boundary can only be drawn there: how dynamo works, where it and the op layer
disagree, and what the boundary costs.

## Calling an op that is already in

### Checking whether an op is in {#supported}

Read the class attribute `compile_op_names`: non-empty means the boundary is at the op
layer and `fullgraph=True` works, an empty tuple means the op has not migrated yet.

```python
>>> from tileops.ops import RMSNormFwdOp
>>> RMSNormFwdOp.compile_op_names
('tileops::norm_rms_norm_fwd',)
```

An op that has not migrated raises under `fullgraph=True`, and breaks the graph under
the default settings.

### Compiling a call

Construct the op instance, hand the function that calls it to `torch.compile`, and
that is all:

```python
import torch
from tileops.ops import RMSNormFwdOp

op = RMSNormFwdOp(normalized_shape=(4096,))     # construct once, reuse

@torch.compile(fullgraph=True)
def block(x, weight):
    return op(x, weight)

x = torch.randn(2048, 4096, device="cuda", dtype=torch.float16)
w = torch.randn(4096, device="cuda", dtype=torch.float16)
block(x, w)
```

`TORCH_LOGS=graph_code` prints the captured graph: one node,
`tileops::norm_rms_norm_fwd`, not the calls inside the kernel.

### The five calling conventions

Each follows from a mechanism at the boundary, and breaking any of them makes the
compiled path behave differently from the eager one.

- **Construct the op instance once and reuse it.** The instance key is a compile-time
  constant and each instance is its own compiled graph, so constructing one inside a loop
  recompiles every iteration.
- **Do not rely on strides passing through.** A non-contiguous input is made contiguous
  inside the node and the output is always contiguous; convert outside the op if later
  work needs another layout.
- **Meta tensors cannot warm anything up.** Once an op has a boundary, a call with
  meta or fake tensors returns at the fake and never reaches kernel construction.
- **Warm up before a CUDA-graph capture.** Call once with real tensors at the same
  shape: building a kernel may compile, while capture allows only a memo hit and the call
  that follows it. See [what each phase may do](backends.md#phase-limits).
- **Every device builds its own kernel.** The device is part of the kernel memo key, so
  the same instance builds again on a second card. A `target=` named in the constructor
  is honoured on the first compiled call too, and a failed build pins the op to no
  target.

### The three guarantees once an op is in

With the boundary at the op layer, a caller can rely on three things.

- **The graph does not change with the target.** The same code compiles to the same
  graph on another backend or another card, so the artefact is independent of who serves
  the op.
- **`fullgraph=True` works**, for an op that declares this contract — see [checking
  whether an op is in](#supported).
- **Output shape, dtype and stride come from the manifest**, not from how a kernel
  tiles or pads internally. Inputs are made contiguous inside the node, and the
  output is always contiguous.

## Declaring the boundary on a new op: `RMSNormFwdOp`

The code bringing one op in takes: how the boundary is declared, how the fake is
written, and why the target is resolved again inside the node. For the tracing, graph
breaks and guards below, see [how dynamo works](#how-dynamo-works).

`RMSNormFwdOp` was the first op brought in. Its skeleton, method bodies elided — the full
file is
[`src/tileops/ops/norm/rms_norm.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/ops/norm/rms_norm.py):

```python
class RMSNormFwdOp(Op):
    # the operators in the graph that belong to this op
    compile_op_names = ("tileops::norm_rms_norm_fwd",)

    def _infer_output_shapes(self, x_shape, weight_shape):
        return {"output": tuple(x_shape)}          # the manifest's shape_rules

    def forward(self, x, weight):
        # the only line: call the opaque operator
        return _rms_norm_fwd(x, weight, self._instance_key)

    def _eager_forward(self, x, weight):
        ...                                        # validate, make contiguous
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

The layers one call passes through, and where the boundary falls:

<figure class="callpath" markdown="0">
  <div class="cp-step cp-traced"><code>Op.__call__</code><span>resolve the target, unsettle on failure</span></div>
  <div class="cp-step cp-traced"><code>forward</code><span>one line, calls the opaque operator</span></div>
  <div class="cp-boundary"><span>compile boundary</span></div>
  <div class="cp-step cp-opaque"><code>_rms_norm_fwd</code><span>the operator body, recovers the instance</span></div>
  <div class="cp-step cp-opaque"><code>_eager_forward</code><span>validate, contiguous, kernel, launch</span></div>
  <figcaption>The two violet layers are inside dynamo's trace, and that one line of <code>forward</code> is the last thing it reaches; below the boundary the opaque operator takes over, invisible to the compiler.</figcaption>
</figure>

Three things in it are not free choices.

**First, the instance is recovered through a string key rather than passed directly.**
The schema's types are a fixed set — `Tensor`, `int`, `float`, `bool`, `str` and a few
more — with no "arbitrary Python object", while what the operator body needs
(`kernel_map`, the settled target, the memo table of built kernels) hangs off the instance
and does not fit a schema argument. Two details of the key are not free either:

- **A string, not an integer.** A string is a compile-time constant during tracing,
  where an integer is generalised to a `SymInt`.
- **Never reused.** Because it is constant, inductor bakes the shape the fake gave
  into the artefact, and an op reusing a key would inherit the previous instance's
  shape.

**Second, the fake builds its result with `x.new_empty(shape)`, not
`torch.empty_like(x)`.** What the fake returns has to match real execution in shape, dtype
and stride; a mismatch either fails during tracing or — for a stride — has downstream code
read the wrong layout and go silently wrong. The operator body makes the inputs contiguous
before the kernel writes into a freshly allocated output, so the real output is always
contiguous, while `empty_like` copies the input's strides: a non-contiguous input would
have the fake declare a layout real execution never produces.

**Third, the target is resolved twice — once in `Op.__call__`, once in
`get_or_build_kernel`.** When traced code runs `self.x = ...`, dynamo records a pending
side effect and applies it only after the whole graph has run, while the opaque node runs
before that: a resolution written just outside the node is unreadable inside it. Two
things follow, both inside the node:

- Without the second resolution, the first compiled call silently runs the wrong
  implementation.
- Undoing a failed resolution is the job of whichever site made it, since a compiled
  artefact does not keep the call site's `try/except`.

All three follow from one fact: torch's compilation and declaration mechanisms
work per function, while what needs compiling is one call on an object.

## Appendix: why the boundary looks like this

### How dynamo works

This section is about how dynamo decides what may enter a graph — which is where
the conditions an op has to satisfy come from.

Dynamo is the front end of `torch.compile`, working at CPython's frame evaluation
layer (PEP 523).

**It has exactly one entry point: `torch.compile`.** `torch.compile(fn)` returns a
wrapper, and tracing happens when that wrapper is called; `nn.Module.compile()`
and the decorator form are two other spellings of the same entry. A call that does
not go through it takes the ordinary Python path and has nothing to do with dynamo
— below, that path is called eager.

On the first call, dynamo takes over the frame, symbolically executes the bytecode
instruction by instruction, records the tensor operations as one FX graph, leaves
what cannot enter the graph in Python, and notes a set of guards for the graph —
the premises this trace relied on, such as a tensor's dtype and rank. Later calls
reuse the compiled artefact when every guard holds; if one fails, the new case is
traced again.

Three terms have fixed meanings on this page:

| Term | Meaning |
| --- | --- |
| Graph | the FX graph dynamo captured; one trace produces one |
| Node | one operator call in the graph, with input edges and the output's shape and dtype |
| Traced | inside dynamo's symbolic execution. Tracing performs no real computation; it records |

The graph then goes to a backend (inductor and others) for fusion, memory planning
and code generation. The larger a graph is, the more neighbouring operators can
fuse, so every op in an operator library has to be able to appear as a node in
someone else's graph.

Two of dynamo's rules matter for bringing an op in:

- **It inlines by default.** A called function is not itself a boundary, and its
  body is folded into the same trace. Keeping a stretch of Python out of the trace
  takes an explicit declaration.
- **Untraceable code is handled one of two ways.** By default dynamo breaks the graph, falling
  back to Python for that stretch, so one graph becomes several; under
  `fullgraph=True` it raises instead. Raising surfaces the problem during
  development, which is why an operator library treats `fullgraph=True` as its
  acceptance criterion.

### Where the op layer and dynamo disagree

Applying those rules to a TileOPs op makes the obstacle plain. Dynamo compiles
frames, that is, functions; a TileOPs op is an object, and one call does four
things, of which only the last belongs in the graph:

| What a call does | Should dynamo capture it |
| --- | --- |
| Validate dtypes and shapes, make the inputs contiguous | No |
| Decide which target serves this call | No |
| Fetch or build the kernel | No — capturing this far fails |
| Launch the kernel and produce the output | Yes, as a node in the graph |

The table needs three qualifications.

**"Should not be captured" is not "does not run."** All four happen on every call;
the only question is what enters the graph.

**The distinction has to be annotated by hand** — dynamo cannot draw it. Torch
provides two interfaces: `torch.library.custom_op` registers the call as an
operator, so dynamo puts a single node in the graph and does not trace into the
implementation, and `register_fake` tells the compiler what the node outputs,
receiving only the inputs' metadata and never touching real data.

**Without the annotation, tracing goes in and fails.** With the boundary
undeclared, `RMSNormFwdOp` compiles in neither of its two states:

- An instance that has not built a kernel builds one during the call, and dynamo
  traces into the TileLang JIT inside the constructor.
- An instance that already has one skips the construction, but still re-parses the
  TileLang program on every call, so dynamo traces into `@tilelang.jit` and stops at
  `inspect.signature`.

### Why the boundary falls at the op layer

The boundary could sit at the op layer, or lower down at the kernel layer. The
difference shows up in the user's compiled graph.

The node's identity in that graph — its name, arguments, granularity, and the
output its fake declares — is the operator the user sees. With the boundary at the
kernel layer, changing backend changes that node, the same op compiles to a
different graph under a different target, and the compiled artefact is tied to the
backend. At the op layer it does not: the node's identity is the op's, independent
of who serves it.

That position also settles how the fake is written. The op layer does not know how
an external kernel tiles or pads internally; the one shape rule that holds for
every target is the one in the manifest, so the fake derives from it.

The node's interior is invisible to the compiler, but its contract to the outside is
complete: the schema gives the name and argument types, the fake gives the output's
shape, dtype, device and stride, and the alias annotations say it does not write to
its inputs. With the contract complete, what is kept and what is given up separate
cleanly:

- **Optimisation between nodes proceeds as usual.** Buffer assignment, lifetimes,
  reordering against neighbours it does not depend on, deletion when nothing
  consumes it.
- **Optimisation inside the node is gone.** Neighbours cannot fuse in, and the
  output must be written to memory.

For an operator library the trade is worth it: inside the node is a kernel TileLang
has already compiled, which inductor need not touch.

### What the boundary costs

Measured on an idle H200 at 2048×4096, fp16. Per-call figures are the minimum of
three runs of 2000 iterations × 9 rounds:

| | Boundary at the kernel layer | Boundary at the op layer |
| --- | --- | --- |
| Kernel time | 0.0119 ms | 0.0117 ms |
| Eager, per call | 42.5–45.2 µs | 38.2–42.0 µs |

The kernel itself is unaffected — where the boundary sits has nothing to do with
how the kernel computes. The eager path is 3–5 µs faster because, with the boundary
moved up, a call crosses one operator boundary instead of two.

The cost on the graph side is in [why the boundary falls at the op
layer](#why-the-boundary-falls-at-the-op-layer): fusion does not cross the node boundary,
and the node's output always lands in memory.

### What the boundary does not provide

| Not provided | Reason |
| --- | --- |
| Fusion across the node boundary | The node's interior is opaque to the compiler, so the elementwise work on either side stays outside |
| autograd through the node | This path serves inference; forward and backward are separate ops |
| Switching target within one compiled artefact | The target belongs to the op instance — another target means another instance, and another graph |
| Building a kernel from meta tensors | A call with meta tensors answers with shapes and dtypes only |
