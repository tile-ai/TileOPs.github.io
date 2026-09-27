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
       dtype_cases: [{T: float16}, {T: bfloat16}], label: ds-v3-prefill-attn-proj}
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
        "gemm_tma_kernel": GemmTmaKernel,
        "gemm_cp_async_kernel": GemmCpAsyncKernel,
        "gemv_kernel": GemvKernel,
    }

    def __init__(self, trans_a=False, trans_b=True, *, target=None, kernel_map=None, tune=False):
        self.trans_a, self.trans_b = trans_a, trans_b
        self.target, self.tune = target, tune
        self.dispatch_kernel(kernel_map)              # installs this instance's kernel map

    def forward(self, a, b):
        return self._call_boundary(a, b)              # the generated operator

    def _eager_forward(self, a, b):                   # the generated checks have run
        a, b = a.contiguous(), b.contiguous()         # handed over as the spec declares it
        m, k = (a.shape[1], a.shape[0]) if self.trans_a else a.shape
        n = b.shape[0] if self.trans_b else b.shape[1]
        kernel = self.kernel_for(
            "gemm",                                   # the memoization bucket
            (a, b),                                   # the tensors the kernel gets
            self._call_spec(m, n, k, a.dtype, a.device),  # what this call is
        )
        return kernel(a, b)
```

| # | Member | Written from |
| --- | --- | --- |
| 1 | `__init__` | the names, order and defaults in `signature.params`, then `target`, `kernel_map` and `tune`, closing with `self.dispatch_kernel(kernel_map)` |
| 2 | `kernel_types` | the Kernel classes that can serve the op, each under a name; a `kernel_map=` override replaces one by that name |
| 3 | `forward` | `signature.inputs` — its order, optional inputs last with default `None` |
| 4 | `_eager_forward` | contiguity, the call record, fetching the kernel and launching it |
| 5 | `compute_roof` | optional: the GPU-profile unit that prices the op's FLOPs, where it is not CUDA-core fp32 |

`_infer_output_shapes`, `_validate_dtypes` and `eval_roofline` are generated from the spec
and are not written.

An op without a compile boundary writes the body of `_eager_forward` in `forward` itself;
declaring the boundary moves it behind the generated operator. How that works is in
[bringing an op into torch.compile](torch-compile.md).

### `kernel_for`, and choosing among kernels

A kernel is a compiled artefact, hundreds of milliseconds to seconds to build, while an op
instance is called over and over at different shapes and dtypes. The op layer therefore
keeps a memo table: a kernel this call needs and has built before comes straight back,
and only otherwise is one built and stored. `kernel_for` is that table's only entrance on
the in-tree path; a [target](backends.md) serves the whole op instead and never reaches it.

Its three arguments:

- **`role`** — the memoization bucket, one per kernel the op runs per call. `GemmFwdOp`
  runs one, so it has one role, whichever of its three classes serves the call.
- **`inputs`** — the tensors the kernel is about to be handed, in `signature.inputs`
  order, one slot per input. An optional input that was not passed keeps its slot as
  `None`.
- **`call`** — what this call is. `GemmCall` carries every fact the GEMM kernels read:
  `m`, `n`, `k`, the dtype, the layout, the device.

Which class serves a call is decided by the classes, not the op. Each states the region
it serves (`applies`, `refusal`), one is marked `general` for everything else, and two
specialised classes claiming one call is an error, never a silent preference. The chosen
class's `entry_for(call)` returns the **identity** two calls must share to reuse one
kernel, and the **builder** that runs once per identity. Carry too little in the identity
and a second dtype reuses the first dtype's kernel; carry the whole shape where the kernel
depends on fewer quantities and it compiles once per distinct shape.

An op with a single kernel and no call record writes `entry_for(role, call)` on the op
itself and states the identity and builder there, as `RMSNormFwdOp` does:

```python
def entry_for(self, role, call):                    # call is the input dtype
    n = math.prod(self.normalized_shape)
    eps = torch.finfo(torch.float32).eps if self.eps is None else float(self.eps)
    return call, lambda: self.kernel_map["rms_norm"](n, eps, call, tune=self.tune)
```

An op with no in-tree implementation, written to depend on a backend, leaves out both
`kernel_types` and `entry_for`; a call on a device no target claims then raises
`OpNotAvailableError`.

### Registering

Add the op's name to the imports and `__all__` in two places: its family's
[`src/tileops/ops/<family>/__init__.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/ops),
where the class is implemented, and
[`src/tileops/<family>.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops), the public path.
Without the second, `from tileops.<family> import ...` will not find the op and the API
reference cannot collect it.

## Step 3: write the kernel

A kernel class subclasses [`Kernel`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/kernels/kernel_base.py), lives under [`src/tileops/kernels/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/kernels), is written in
TileLang, compiles at construction and launches on `__call__`. Its constructor is what its
`entry_for` builder calls, and its call signature is the `kernel(a, b)` of step 2.

This is the one place of the six the spec does not constrain: a kernel neither reads the
spec nor is checked against it.

How the constructor and the call divide their arguments is a hard requirement: **only
values compiled into the generated code go in the constructor.** `GemmTmaKernel` divides
them like this:

```python
class GemmTmaKernel(Kernel):
    def __init__(self, m, n, k, dtype, config=None, tune=False, trans_a=False, trans_b=False, ...):
        self.kernel = _gemm_kernel(m, n, k, trans_a, trans_b, self.dtype_str, ...)  # compiles
        self.init_config(config, tune)      # tile sizes and pipeline depth

    def __call__(self, a, b):               # a call passes tensors, nothing else
        ...
```

`m`, `n`, `k`, the dtype and the two layout flags are constructor arguments because the
generated code treats them as constants: loop bounds, TMA descriptors and the WGMMA shape
all unroll from them, as do the tile sizes. The tensors belong to `__call__`, where each
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
branches — small shapes marked `smoke` for the PR checks, large ones `full` for the
nightly. The workload rows are not unit-test coverage; the
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
