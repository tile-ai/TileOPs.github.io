# Adding a new op

A new op means writing code in the six places below, and the table is in the order to work
through them.

The spec goes first: it decides what the other five files contain, and in the end it is what
they are checked against. **The spec is this pipeline's input, and the other five are written
from it.**{ .keystone }

| # | File | Held to the spec by | Contents |
| --- | --- | --- | --- |
| 1 | [`src/tileops/manifest/spec/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/manifest/spec)`<family>.yaml` | the validator's `schema` and `signature` levels | the spec itself |
| 2 | [`src/tileops/ops/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/ops)`<family>/…` | the validator, against `__init__`, `forward` and the declared kernels; the checks generated around every call | the op class, subclassing `Op` |
| 2 | [`src/tileops/ops/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/ops)`<family>/__init__.py` and [`src/tileops/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops)`<family>.py` | the validator: the family's `__all__` agrees with the manifest | the op's name, exported by its family and on the public path `tileops.<family>.<Op>` |
| 3 | [`src/tileops/kernels/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/kernels)`<family>/…` | — | the kernel classes, subclassing `Kernel` |
| 4 | [`tests/ops/`](https://github.com/tile-ai/TileOPs/tree/main/tests/ops)`test_<name>.py` | the contract tests, which run every workload row | the comparison against the reference `ref_program` |
| 5 | [`benchmarks/ops/`](https://github.com/tile-ai/TileOPs/tree/main/benchmarks/ops)`bench_<name>.py` | the validator's `bench` level | the benchmark |

`GemmFwdOp` — the plainest matmul there is — runs through all six below.

## Step 1: write the spec

What the fields mean and how to write them is in [writing a spec](manifest.md). A new op
starts at `status: spec-only`: the interface is settled and there is no implementation
yet, so the checks that read code are skipped and do not fail over the missing class.

`GemmFwdOp`'s spec, with one of its workload rows:

```yaml
GemmFwdOp:
  ref_api: torch.matmul
  family: gemm
  status: spec-only
  signature:
    types:
      Mat:
        params: {t: Bool, R: Dim, C: Dim}
        match: t
        cases:
          - {when: false, is: "[R, C]"}
          - {when: true, is: "[C, R]"}
    forall: {M: Dim, N: Dim, K: Dim, T: "DType[float16 | bfloat16]"}
    params:
      trans_a: {type: bool, default: false}
      trans_b: {type: bool, default: true}
    inputs:
      a: {dtype: T, shape: "Mat[trans_a, M, K]"}
      b: {dtype: T, shape: "Mat[trans_b, K, N]"}
    outputs:
      d: {dtype: T, shape: "[M, N]"}
  workloads:
    - {M: 4096, N: 4096, K: 7168, trans_a: false, trans_b: true,
       dtype_cases: [{T: float16}, {T: bfloat16}], label: ds-v3-prefill-mlp-up}
  roofline:
    flops: "2 * M * N * K"
```

The spec names no file and no kernel: which kernels serve the op is a fact of the code,
declared on the op class in step 2.

## Step 2: write the op class {#op-class}

The op class subclasses [`Op`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/ops/op_base.py) and sits between the spec and the kernel. The checks
around every call — dtypes, shapes, the refinements, output shape inference — are
generated from the signature when the class is defined, so the class writes none of them.
What it writes is how a call reaches a kernel.

### The class, and its members

`GemmFwdOp`'s skeleton, docstrings elided:

```python
class GemmFwdOp(Op):
    compile_boundary: ClassVar[bool] = True           # optional: claims fullgraph=True
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gemm_tma": GemmTmaKernel,
        "gemm_cp_async": GemmCpAsyncKernel,
        "gemv": GemvKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"gemm": GemmFwdInterface}

    def __init__(self, trans_a=False, trans_b=True, *, target=None, kernel_map=None, tune=False):
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.target = target
        self.tune = tune
        self.dispatch_kernel(kernel_map)              # installs this instance's kernel map

    def forward(self, a, b):
        return self._call_boundary(a, b)              # the generated operator

    def _eager_forward(self, a, b):                   # the generated checks have run
        a, b = a.contiguous(), b.contiguous()         # handed over as the spec declares it
        m, k = (a.shape[1], a.shape[0]) if self.trans_a else a.shape
        call = GemmCall(                              # what this call is
            m=m,
            n=b.shape[0] if self.trans_b else b.shape[1],
            k=k,
            dtype=a.dtype,
            trans_a=self.trans_a,
            trans_b=self.trans_b,
            device=a.device,
        )
        return self.kernel_for("gemm", call)(a, b)
```

| # | Member | Written from |
| --- | --- | --- |
| 1 | `__init__` | the names, order and defaults in `signature.params`, then `target`, `kernel_map` and `tune`, closing with `self.dispatch_kernel(kernel_map)` |
| 2 | `kernel_types` | the Kernel classes that can serve the op, each under a key; a `kernel_map=` override replaces one by that key |
| 3 | `interfaces` | one entry per kernel call the op makes: the name `kernel_for` uses → the `KernelInterface` class whose implementations serve that call |
| 4 | `forward` | `signature.inputs` — its order, optional inputs last with default `None` |
| 5 | `_eager_forward` | contiguity, the call spec, fetching the kernel and launching it |
| 6 | `compute_roof` | optional: the GPU-profile unit that prices the op's FLOPs, where it is not CUDA-core fp32 |

`_infer_output_shapes`, `_validate_dtypes` and `eval_roofline` are generated from the spec
and are not written.

An op without a compile boundary writes the body of `_eager_forward` in `forward` itself;
declaring the boundary moves it behind the generated operator. How that works is in
[bringing an op into torch.compile](torch-compile.md).

### `kernel_for`, and choosing among kernels {#kernel-selection}

A kernel is a compiled artefact, hundreds of milliseconds to seconds to build, while an op
instance is called over and over at different shapes and dtypes. The op layer therefore
keeps a memo table: a kernel this call needs and has built before comes straight back,
and only otherwise is one built and stored. `kernel_for` is that table's only entrance on
the in-tree path; a [target](backends.md) serves the whole op instead and never reaches it.

Its two arguments:

- **`interface`** — a key of `interfaces`, naming one kernel call the op makes.
  `GemmFwdOp` makes one, so it declares one, `"gemm"`. A second interface is opened only where the semantics or the call contract changes:
  `BatchNormFwdOp` has `batch_norm_fwd_train` and `batch_norm_fwd_infer`, which return
  different things. A faster kernel for some shape range or some architecture is another
  implementation of the interface already there.
- **`call`** — a frozen `CallSpec` subclass carrying the facts needed to select and build
  the kernel: shapes, the dtype, the op's semantic parameters, and the device. It
  has to be the interface's `request` type. The dispatcher derives the device facts (`arch`,
  `sm_count`, `calibration`, `smem_budget`) itself, from
  `call.device` on a miss.

The kernel that comes back is called with the parameters of the interface's abstract
`forward`, in that order.

An interface is a class in
[`src/tileops/kernels/<family>/call_spec.py`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/kernels),
beside the call spec it names in `request`; a family with one kernel file keeps both in
that file. Its name is `{Name}{Fwd|Bwd}Interface`, with variant words before the
direction, which `interface-names-lint` checks. Its abstract `forward` is the whole contract
an implementation — in-tree or from a backend — is written against, so its docstring
states each tensor's shape, dtype, layout, device and whether it is written in place:

```python
class GemmFwdInterface(KernelInterface):
    """Dense matmul under the ``(trans_a, trans_b)`` layout the call states."""

    request = GemmCall

    @abstractmethod
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiply the two matrices; nothing is written in place.

        Both operands are contiguous on ``call.device`` in ``call.dtype``, which is
        ``float16`` or ``bfloat16``; the contraction accumulates in ``float32``.

        Args:
            a: ``(call.m, call.k)``, or ``(call.k, call.m)`` when ``call.trans_a``.
            b: ``(call.n, call.k)`` when ``call.trans_b``, else ``(call.k, call.n)``.

        Returns:
            A new ``(call.m, call.n)`` tensor in ``call.dtype``.
        """
```

An implementation is a class inheriting both `Kernel` and one interface, listed in
`kernel_types` under a key. Which implementation serves a call follows from four
declarations the implementations make about themselves; the op makes none of them:

| # | Declaration | States | Left undeclared |
| --- | --- | --- | --- |
| 1 | `devices`, `supported_archs` | where the implementation runs | CUDA devices, every architecture |
| 2 | `applies(call)`, `refusal(call)` | which calls it serves, stated positively | every call |
| 3 | `general`, `preferred_over` | which implementation wins where two of them serve one call | wins over none |
| 4 | `entry_for(call)` | the build identity, and the builder that runs once per identity | the whole call spec, built by `cls(call)` |

The dispatcher filters by availability first, then picks the single winner among the
implementations that are left and that apply: `general` loses to every other one, and the
rest are compared by the keys each names in `preferred_over`. Nothing left raises
`no implementation serves this call`, or `OpNotAvailableError` where no key runs on the
call's device type at all; two with no relation between them raise `dispatch is
ambiguous`. Declaration order decides nothing. Where one implementation should give up a
range to another, the one that should win declares `preferred_over`, rather than the
other one excluding that range in its own `applies`.

`GemmFwdOp`'s three implementations cover the `"gemm"` interface's calls like this:

| # | Key | Serves | Declares |
| --- | --- | --- | --- |
| 1 | `gemm_tma` | SM90 shapes whose operands TMA can address | `supported_archs = [90]`, and a `refusal` naming the misalignment |
| 2 | `gemv` | shapes of at most two rows contracted over K, where reducing on CUDA cores wins | `supported_archs = [90]`, `applies` through `band_for`, `preferred_over = frozenset({"gemm_tma"})` |
| 3 | `gemm_cp_async` | every shape the other two do not claim, provided a K row spans at least one four-byte load | `supported_archs = [80, 86, 89, 90]`, `general = True`, and a `refusal` for a narrower K row |

`entry_for(call)` returns the **identity** two calls must share to reuse one kernel, and
the **builder** that runs once per identity. Carry too little in the identity and a second
dtype reuses the first dtype's kernel; carry the whole shape where the kernel depends on
fewer quantities and it compiles once per distinct shape.

An interface with one implementation needs nothing beyond inheriting it. `RMSNormKernel`
is the only implementation `RMSNormFwdOp` has
([`src/tileops/kernels/norm/rms_norm.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/kernels/norm/rms_norm.py)):

```python
class RMSNormKernel(Kernel, RMSNormFwdInterface):
    supported_archs: list[int] = [80, 86, 89, 90]

    @classmethod
    def entry_for(cls, call: LayerNormCall) -> Entry:
        identity = (call.n, call.eps, call.dtype)
        return identity, lambda: cls(*identity)
```

An op defines no `entry_for` of its own and keeps no kernel cache of its own — no dict, no
build guarded on an attribute being unset. Holding what `kernel_for` returned in
`self.kernel` is not one.

An op with no in-tree implementation, written to depend on a backend, declares neither
`kernel_types` nor `interfaces`; a call on a device no target claims then raises
`OpNotAvailableError`.

A backend adds an implementation to an interface, or replaces the class registered under
one key,
without changing TileOPs; both are in [adding a hardware backend](backends.md).

### Registering

Add the op's name to the imports and `__all__` in two places: its family's
[`src/tileops/ops/<family>/__init__.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/ops),
where the class is implemented, and
[`src/tileops/<family>.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops), the public path.
Without the second, `from tileops.<family> import ...` will not find the op and the API
reference cannot collect it.

## Step 3: write the kernel

A kernel class subclasses [`Kernel`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/kernels/kernel_base.py) and the interface it implements, lives under [`src/tileops/kernels/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/kernels), is written in
TileLang, compiles at construction and implements `forward`, which the base class's
`__call__` runs. Its constructor is what its own `entry_for` builder calls, and its
`forward` takes the interface's parameters, the `kernel(a, b)` of step 2.

This is the one place of the six the spec does not constrain: a kernel neither reads the
spec nor is checked against it.

How the constructor and the call divide their arguments is a hard requirement: **only
values compiled into the generated code go in the constructor.** `GemmTmaKernel` divides
them like this:

```python
class GemmTmaKernel(Kernel, GemmFwdInterface):
    def __init__(self, m, n, k, dtype, config=None, tune=False, trans_a=False, trans_b=False, ...):
        self.kernel = _gemm_kernel(m, n, k, trans_a, trans_b, self.dtype_str, ...)  # compiles
        self.init_config(config, tune)      # tile sizes and pipeline depth

    def forward(self, a, b):                # a call passes tensors, nothing else
        ...
```

`m`, `n`, `k`, the dtype and the two layout flags are constructor arguments because the
generated code treats them as constants: loop bounds, TMA descriptors and the WGMMA shape
all unroll from them, as do the tile sizes. The tensors belong to `forward`, where each
call swaps pointers.

Dividing them wrong costs a recompile. A decode step advances one token at a time, so
`seq_len` grows by one every step and batch changes with the running set:

```python
# wrong: seq_len in the constructor — every step is a new kernel
kernel = AttnKernel(batch, seq_len, num_heads, dtype)

# right: compile-time constants in the constructor, the varying sizes per call
kernel = AttnKernel(num_heads, head_dim, dtype)
out = kernel(q, k, v)                       # seq_len is read off the tensor shapes
```

With the first form, `seq_len` ends up in the identity `entry_for` returns, every step
misses, every step compiles, and decode goes nowhere.

## Step 4: write the test

Tests live in [`tests/ops/`](https://github.com/tile-ai/TileOPs/tree/main/tests/ops) and compare against `ref_program`, the reference the
workload (or the test class) defines, over shapes the test chooses to reach the kernel's
branches. Cases marked `smoke` run on every PR. Cases marked `full` run on any PR that
changes their test file, and on the nightly. Long-running cases are marked `nightly` and
run only on the nightly. The workload rows are not unit-test coverage; the
contract tests already run each of them through the op.

The scaffolding is `TestBase` and `FixtureBase` from
[`tests/test_base.py`](https://github.com/tile-ai/TileOPs/blob/main/tests/test_base.py), with the cases in `PARAMS`.

Where the op has an optional input, both sides need a case — passed and not passed often
run different kernels.

## Step 5: write the benchmark

Benchmarks live in [`benchmarks/ops/`](https://github.com/tile-ai/TileOPs/tree/main/benchmarks/ops) and time each call through a `ManifestBenchmark`
built around the op and the call's workload. The calls are not written here:
`manifest_calls(<Op>)` instantiates each workload row with each of its dtype cases and ids
the case by its case id, and the validator's `bench` level fails a benchmark that writes
its own:

```python
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops import GemmFwdOp
from workloads.gemm import GemmWorkload


@pytest.mark.parametrize("call", manifest_calls(GemmFwdOp))
def test_gemm_bench(call) -> None:
    workload = GemmWorkload.from_call(call)
    a, b = workload.gen_inputs()
    op = GemmFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, workload)
    bm.compare({"tileops": op, "torch-cublas": workload.ref_program}, a, b)
```

Record at least one non-TileOPs baseline as well, or the row has nothing to compare
against. Where a baseline needs its input converted, that conversion stays inside its own
timed region. What the reported numbers mean is in [how a benchmark is timed](timing.md).

## Step 6: flip the status, and let CI take over

With the other five written, check your own work with the three commands below:

```bash
python scripts/validate_manifest.py --check-op GemmFwdOp   # spec and code agree
python -m pytest tests/ops/test_gemm.py -v                # numerics match ref_program
python -m pytest benchmarks/ops/bench_gemm.py             # the benchmark produces numbers
```

With all three passing, flip the spec's `status` from `spec-only` to `implemented`. That
one edit turns on the checks that read code and puts the op inside CI's reach: every later
change is held against the spec by the validator, the tests and the nightly benchmark.

## Afterwards

Once the op runs, two optional things remain:

- Let the op into a user's compiled graph — [bringing an op into
  torch.compile](torch-compile.md).
- Let someone else's kernels serve it on other hardware — [adding a hardware
  backend](backends.md).
