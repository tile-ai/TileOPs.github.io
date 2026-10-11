# Adding a new op

Adding an op means writing code in the six places below. The order of the table is
also the recommended order of writing.

The spec is written first, because it decides what the other five files contain, and
in the end they are checked against it. **The spec is the input to this workflow, and
the other five places are written from it.**{ .keystone }

| # | File | Held to the spec by | Contents |
| --- | --- | --- | --- |
| 1 | [`src/tileops/manifest/spec/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/manifest/spec)`<family>.yaml` | the validator's `schema` and `signature` levels | the spec itself |
| 2 | [`src/tileops/ops/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/ops)`<family>/…` | the validator, against `__init__`, `forward` and the declared kernels; the checks generated around every call | the op class, subclassing `Op` |
| 2 | [`src/tileops/ops/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/ops)`<family>/__init__.py` and [`src/tileops/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops)`<family>.py` | the validator: the family's `__all__` agrees with the manifest | the op's name, exported by its family and on the public path `tileops.<family>.<Op>` |
| 3 | [`src/tileops/kernels/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/kernels)`<family>/…` | — | the kernel classes, subclassing `Kernel` |
| 4 | [`tests/ops/`](https://github.com/tile-ai/TileOPs/tree/main/tests/ops)`test_<name>.py` | the contract tests, which run every workload row | the numerical comparison against the reference `ref_program` |
| 5 | [`benchmarks/ops/`](https://github.com/tile-ai/TileOPs/tree/main/benchmarks/ops)`bench_<name>.py` | the validator's `bench` level | the benchmark, and the op's case entry in `benchmarks/_cases/` |

The steps below take `GemmFwdOp`, the simplest matmul, through all six places.

## Step 1: write the spec

What each field means and how to write it is in
[Reading and writing the manifest](user-guide/manifest/index.md). A new op starts at
`status: spec-only`: the interface is settled and the implementation is not written
yet. The checks that read code are skipped at this status, so they do not fail on the
missing class.

`GemmFwdOp`'s spec, with one workload row:

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

The spec names no file and no kernel. Which kernels serve the op belongs to the code,
and is declared on the op class in step 2.

## Step 2: write the op class {#op-class}

The op class subclasses [`Op`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/ops/op_base.py) and sits between the spec and the kernel. The checks
around every call (dtypes, shapes, the refinements and output shape inference) are
generated from the signature when the class is defined, and the class writes none of
them. What the class writes is how a call reaches a kernel.

### The class, and its members

`GemmFwdOp`'s skeleton, with docstrings elided:

```python
class GemmFwdOp(Op):
    kernel_types: ClassVar[Mapping[str, type[Kernel]]] = {
        "gemm_tma": GemmTMAKernel,
        "gemm_cp_async": GemmCpAsyncKernel,
        "gemv": GemvKernel,
    }
    interfaces: ClassVar[Mapping[str, type[KernelInterface]]] = {"gemm": GemmFwdInterface}

    def __init__(self, trans_a=False, trans_b=True, *, target=None):
        self.trans_a = trans_a
        self.trans_b = trans_b
        super().__init__(target=target)               # checks the params, installs kernel_types

    def forward(self, a, b):                          # the generated checks have run
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

    def roof_key(self):                               # FLOPs priced on tensor cores
        return tensor_core_roof(self.last_call.indices["T"])
```

| # | Member | Written from |
| --- | --- | --- |
| 1 | `__init__` | the names, order and defaults in `signature.params`, then the keyword-only `target`; it assigns each manifest parameter to the attribute of the same name, then calls `super().__init__(target=target)` |
| 2 | `kernel_types` | the Kernel classes that can serve the op, each under a key |
| 3 | `interfaces` | one entry per kernel call the op makes, mapping the name `kernel_for` uses to the `KernelInterface` class that the implementations serving that call inherit |
| 4 | `forward` | the parameters: the order of `signature.inputs`, with optional inputs last and defaulting to `None`; the body: making the inputs contiguous, building the call spec, fetching the kernel and calling it |
| 5 | `roof_key` | optional: the hardware unit whose peak prices the op's FLOPs; the default is fp32 on CUDA cores, and the member is written only when the op uses another unit |

`_infer_output_shapes` and `eval_roofline` are generated from the spec and are not
written by hand.

`forward` is the op's computation and the only method that carries it. A caller writes
`op(a, b)`, never `op.forward(a, b)`: `Op.__call__` runs the generated checks and then
`forward`. An op whose spec has a call-time tensor input and no `composition` gets a
compile boundary generated for it, and `__call__` runs `forward` behind the generated
operator. How that works is in [Bringing an op into torch.compile](torch-compile.md).

The constructor takes no tuning parameter. To tune, construct the op, then call
`op.request_tune()`.

### `kernel_for`, and choosing among kernels {#kernel-selection}

A kernel is a compiled artefact that takes hundreds of milliseconds to seconds to
build, while an op instance is called many times at different shapes and dtypes. The
op layer therefore keeps a cache table: a kernel this call needs that has been built
before is returned from the table, and otherwise the kernel is built and stored.
`kernel_for` is the only way in-tree implementations reach that table; a
[target](backends.md) serves the whole op and does not go through `kernel_for`.

`kernel_for` takes two arguments:

- **`interface`**: a key of `interfaces`, naming one kernel call the op makes.
  `GemmFwdOp` makes one kernel call, so it declares one interface, `"gemm"`. A second
  kernel interface is added only where the semantics or the call contract changes:
  `BatchNormFwdOp` has `batch_norm_fwd_train` and `batch_norm_fwd_infer`, which return
  different things. A kernel that is faster on some shape range or some architecture
  is another implementation of the existing kernel interface.
- **`call`**: a frozen `CallSpec` subclass carrying the facts needed to select an
  implementation and build the kernel: shapes, the dtype, the op's semantic parameters,
  and the device. It must be the type the kernel interface names in `request`. The
  dispatcher derives the device facts (`arch`, `sm_count`, `calibration`,
  `smem_budget`) from `call.device` on a cache miss; the caller does not fill them in.

The op calls the returned kernel with the parameters of the kernel interface's
abstract `forward`, in that order.

A kernel interface is a class in
[`src/tileops/kernels/<family>/call_spec.py`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/kernels),
next to the call spec it names in `request`; a family with one kernel file keeps both
in that file. Its name has the form `{Name}{Fwd|Bwd}Interface`, with variant words
before the direction, and `interface-names-lint` checks it. Its abstract `forward` is
the only contract every implementation, in-tree or from a backend, is written against,
so its docstring states each tensor's shape, dtype, memory layout, device, and whether
it is written in place:

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

An implementation is a class that inherits both `Kernel` and one kernel interface, and
is listed in `kernel_types` under a key. Which implementation serves a call follows
from the availability, applicability and precedence each implementation declares about
itself; the op takes no part in the selection. A kernel interface with one
implementation needs no declaration beyond the inheritance. The selection rules, how to
write each declaration, and the common errors are in
[How an op selects a kernel](user-guide/dispatch/index.md) and
[Adding a kernel to an op](user-guide/dispatch/writing.md).

An op does not write `entry_for` and keeps no kernel cache of its own. An op with no
in-tree implementation, which depends only on an external backend, declares neither
`kernel_types` nor `interfaces`; when no target claims the call's device, the call
raises `OpNotAvailableError`. How a backend adds an implementation or serves the whole
op is in [How a backend joins TileOPs](user-guide/dispatch/backends.md).

### Registering

Add the op's name to the imports and `__all__` in two places:

1. its family's [`src/tileops/ops/<family>/__init__.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/ops), where the class is implemented;
1. [`src/tileops/<family>.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops), the public path.

Without the second, `from tileops.<family> import ...` does not find the op, and the
API reference does not include it.

## Step 3: write the kernel

A kernel class subclasses [`Kernel`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/kernels/kernel_base.py) and the kernel interface it implements, lives under [`src/tileops/kernels/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/kernels), is written in
TileLang. Construction obtains the TileLang builder; the program compiles the first
time a call runs it with a given config. It implements `forward`, which the base
class's `__call__` runs. Its constructor is called by the builder its own `entry_for`
returns, and its `forward` takes the kernel interface's parameters, the `kernel(a, b)`
of step 2.

The kernel is the only one of the six places the spec does not constrain: a kernel
neither reads the spec nor is checked against it.

The split of arguments between the constructor and the call is a hard requirement:
**only values compiled into the generated code go in the constructor.** `GemmTMAKernel`
splits them like this:

```python
class GemmTMAKernel(Kernel, GemmFwdInterface):
    def __init__(self, m, n, k, dtype, config=None, trans_a=False, trans_b=False, ...):
        self.kernel = _gemm_kernel(m, n, k, trans_a, trans_b, self.dtype_str, ...)  # the TileLang builder
        self.init_config(config)            # tile sizes and pipeline depth

    def forward(self, a, b):                # a call passes tensors, nothing else
        ...
```

`m`, `n`, `k`, the dtype and the two layout flags are constructor arguments because
the generated code treats them as constants: loop bounds, TMA descriptors and the WGMMA
shape are all unrolled from them, and so are the tile sizes. The tensors are passed to
`forward`, and each call only changes the pointers.

A wrong split causes recompilation. Decode advances one token per step, so `seq_len`
grows by one every step, and the batch changes with the running set:

```python
# wrong: seq_len in the constructor — every step is a new kernel
kernel = AttnKernel(batch, seq_len, num_heads, dtype)

# right: compile-time constants in the constructor, the varying sizes per call
kernel = AttnKernel(num_heads, head_dim, dtype)
out = kernel(q, k, v)                       # seq_len is read off the tensor shapes
```

With the first form, the build identity `entry_for` returns contains `seq_len`, so
every step misses the cache and compiles once, and decode cannot run at a usable
speed.

## Step 4: write the test

Tests live in [`tests/ops/`](https://github.com/tile-ai/TileOPs/tree/main/tests/ops) and compare against `ref_program`, the reference
implementation the workload (or the test class) defines. Each test chooses its own
shapes to cover the kernel's branches. Cases fall into three groups by when they run:

- cases marked `smoke` run on every PR;
- cases marked `full` run on any PR that changes their test file, and in the nightly;
- long-running cases are marked `nightly` and run only in the nightly.

The workload rows are not part of unit-test coverage, because the contract tests
already run each row through the op.

The test scaffolding is `TestBase` from
[`tests/workload_test_base.py`](https://github.com/tile-ai/TileOPs/blob/main/tests/workload_test_base.py), with the cases in `PARAMS`.

When the op has an optional input, it needs at least one case with the input passed
and one without, because the two often run different kernels.

An op with a compile boundary, that is, one whose spec has a call-time tensor input and
no `composition`, also needs a cold `torch.compile(op, fullgraph=True)` test, registered
through `register_compile_contract` in
[`tests/compile_contract.py`](https://github.com/tile-ai/TileOPs/blob/main/tests/compile_contract.py).
A test compares the registered ops with the implemented ones that have a compile
boundary, and fails on any difference.

## Step 5: write the benchmark

Benchmarks live in [`benchmarks/ops/`](https://github.com/tile-ai/TileOPs/tree/main/benchmarks/ops). The cases are not written by hand:
`bench.cases(<Op>)` turns each workload row, with each of its dtype cases, into one
`bench.Case` named by its case id, and `bench.Runner(op, case).compare()` checks the op
and every implementation it is compared with against the case's reference, then times
them and records the results. The validator's `bench` level checks that every benchmark
file calls both:

```python
import pytest

from benchmarks import api as bench
from tileops.ops import GemmFwdOp


@pytest.mark.parametrize("case", bench.cases(GemmFwdOp), ids=lambda case: case.id)
def test_gemm_bench(case) -> None:
    op = GemmFwdOp(**case.arguments)
    bench.Runner(op, case).compare({"torch-cublas": case.reference})
```

`bench.cases()` builds a case through the op's entry in
[`benchmarks/_cases/`](https://github.com/tile-ai/TileOPs/tree/main/benchmarks/_cases),
which says how a manifest call becomes a workload. A new op adds its entry to its
family's module there; without one, pytest fails while collecting the cases.

A benchmark also records at least one non-TileOPs implementation; without one, the row
has nothing to compare against. Where an implementation needs its input converted, the
conversion stays inside its own timed region. How to pass implementations that need
private arguments, and the order in which they are checked and timed, are in
[Writing benchmarks](user-guide/benchmark/writing.md); what the reported numbers mean
is in [How a benchmark is timed](timing.md).

## Step 6: flip the status, and let CI take over

With the other five places written, run the three commands below to check the work:

```bash
python scripts/validate_manifest.py --check-op GemmFwdOp   # spec and code agree
python -m pytest tests/ops/test_gemm.py -v                # numerics match ref_program
python -m pytest benchmarks/ops/bench_gemm.py             # the benchmark produces numbers
```

When all three pass, change the spec's `status` from `spec-only` to `implemented`.
That one edit turns on the checks that read code and brings the op under CI: from then
on, the validator, the tests and the nightly benchmark check every change against the
spec.

## Afterwards

Once the op runs, letting other kernels serve it on other hardware is optional; see
[Adding a hardware backend](backends.md). How the op behaves inside a user's compiled
graph is in [Bringing an op into torch.compile](torch-compile.md).
