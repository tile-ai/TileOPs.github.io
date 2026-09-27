# Reading and writing an op's spec

A conventional operator library is organised around its implementations: kernels are
written and tuned one at a time, and what shapes and dtypes each supports, and how fast it
runs, gets described afterwards.

TileOPs is organised the other way round: an op's specification is declared first, and
the implementation is derived from it. That specification is the op's **spec**, a YAML
entry under
[`src/tileops/manifest/spec/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/manifest/spec),
in the file named after its family; those files together are the manifest.

**A spec makes the op an input to the whole system.** Every stage reads the same
declaration rather than reading the implementation:

| Consumer | Reads from the spec | Produces |
| --- | --- | --- |
| The op layer | `signature` | the checks around every call, output shape inference, the dtype check, and the operator `torch.compile` sees |
| [The contract tests](https://github.com/tile-ai/TileOPs/tree/main/tests) | `workloads` | one call per workload row and dtype case, run through the op |
| [The nightly benchmark](https://github.com/tile-ai/TileOPs/tree/main/benchmarks) | `workloads` | the device time of each of those calls |
| [Roofline](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/perf) | `roofline` | the FLOPs and bytes one call moves — the denominator of efficiency |
| This site | `signature`, `workloads` | the shapes printed under each row of the Benchmarks pages |
| CI's [spec validator](https://github.com/tile-ai/TileOPs/blob/main/scripts/validate_manifest.py) | every field | the check that declaration and implementation agree — see [the spec validator](#spec-validator) |

**Every row presupposes a spec**: without one there is no generated validation, no
contract test, no performance data, and nothing in CI holding a regression back.
**Writing a spec is not documenting the op; it is connecting the op to that flow.**{ .keystone }

This page covers what an entry contains, how to read one, how to write one in five steps,
four forms that recur across the manifest, and what the validator checks. The complete
rules are in [Op Manifest](design/manifest.md); this page does not restate them.

## What an entry contains

Each family file is a mapping `op name → entry`; a large family shards into
`<family>_<shard>.yaml`. The files merge at load time, and a duplicate op name is an
error. Algebraic data types that several entries share live in `spec/types.yaml`.

The key is the op's Python class name, `{Name}[{Fwd|Bwd}]Op` — the direction suffix is
required once the other direction also has an entry — and the validator requires
`cls.__name__` to equal it character for character.

| Field | Required | Contents |
| --- | --- | --- |
| `family` | yes | the public module: the op is importable as `tileops.<family>.<Op>` |
| `status` | yes | `implemented`, or `spec-only` while no conforming implementation exists |
| `ref_api` | no | the qualified name of the API the op follows, e.g. `torch.nn.functional.rms_norm` |
| `signature` | yes | the op's type, below |
| `workloads` | yes | the calls tests and benchmarks run |
| `roofline` | yes | the cost of one call, specified in [Roofline](design/roofline.md) |
| `composition` | no | for a composite op, its stages in order: the sub-op classes it may hold and its own kernel roles |

The signature is a function type over named type indices:

| Sub-field | Contents |
| --- | --- |
| `forall` | every free index with its kind: `Dim` (an axis length), `Shape` (a tuple of axes), `DType[...]` (one of a set of dtypes), `Seq[Int]` (a value list only a generator takes) |
| `params` | the `__init__` parameters: `type`, optional `default` and `kw_only` |
| `inputs` / `outputs` | tensors, each `{dtype, shape}` written in those indices, plus presence and effect flags |
| `types` | type families: a shape chosen by the value of a flag |
| `let` | quantities derived from the indices |
| `shape_rules` | refinements: predicates on index values |
| `dtype_combos` | the supported combinations of several `DType` indices, where not every one works |

Key order is position — `params` in `__init__`, `inputs` in `forward`, `outputs` in the
returned tuple — so reordering is a breaking change.

## Reading a spec

`RMSNormFwdOp`, with two of its eight workload rows:

```yaml
RMSNormFwdOp:
  ref_api: torch.nn.functional.rms_norm
  family: norm
  status: implemented
  signature:
    forall: {B: Shape, T: "DType[float16 | bfloat16]"}
    params:
      normalized_shape: {type: "list[int] | tuple[int, ...]"}
      eps: {type: "float | None", default: null}
    inputs:
      x: {dtype: T, shape: "[*B, *normalized_shape]"}
      weight: {dtype: T, shape: "[*normalized_shape]", optional: true}
    outputs:
      output: {dtype: T, shape: "[*B, *normalized_shape]"}
    shape_rules:
      - "len(normalized_shape) > 0"
  workloads:
    - {B: [2048], normalized_shape: [4096], eps: 1.0e-06, some: [weight],
       dtype_cases: [{T: float16}, {T: bfloat16}], label: llama-8b-prefill}
    - {B: [4, 2048], normalized_shape: [128], dtype_cases: [{T: bfloat16}], label: qk-norm-head}
  roofline:
    flops: "(4 if present(weight) else 3) * prod(B) * prod(normalized_shape)"
```

Read it in five steps:

1. **`forall`** — what varies between calls. `B` is the leading axes of any rank, and
   `T` the dtype.
2. **`inputs` and `outputs`** — a shared name is an equality. `x`, `weight` and `output`
   all have dtype `T`, so the three agree; `x` and `output` have one shape term, so they
   have one shape. `weight` is `optional: true`, so a call may omit it.
3. **`params`** — `normalized_shape` is a construction argument that also appears in a
   shape, spliced with `*`: the trailing axes of `x` must equal it.
4. **`shape_rules`** — what the shapes cannot say. Here, that at least one axis is
   normalized.
5. **`workloads`** — each row with each of its `dtype_cases` is one call. A row gives the
   indices (`B`), the construction arguments, and in `some` the optional tensors it
   passes. Its case id is the label followed by the dtype values, `llama-8b-prefill-float16`:
   that is the id a benchmark row carries on the nightly and on this site.

Reading specs programmatically:

```python
from tileops.manifest import load_manifest, load_workloads

ops = load_manifest()                      # every entry, merged
list(ops["RMSNormFwdOp"]["signature"]["inputs"])  # ['x', 'weight']
load_workloads("RMSNormFwdOp")             # that op's workload rows
```

## Writing a spec {#writing-a-spec}

Five steps, each one checkable immediately.

1. **Name it and pick the family.** The key is the class name, and the entry goes in
   `spec/<family>.yaml`.
2. **Write the signature.** Declare each axis length, shape and dtype that varies in
   `forall`, and write every tensor's `dtype` and `shape` in those indices. `params`
   is the op's `__init__` parameter list, less the execution-policy parameters the code
   owns (`target`, `kernel_map`, `tune`). Optional inputs come after the required ones.
   Declare what the reference API supports, not what the current kernel does.
3. **Write the refinements.** `shape_rules` hold predicates on index values, such as
   `H % G == 0`; a derived quantity is a `let`; a shape a flag chooses is a type family.
   A rule never reads a tensor (`x.shape`, `x is None`): presence is `present(x)`.
4. **Write `workloads`.** Each row gives exactly the indices no generator determines,
   every construction parameter without a default, `some` for the optional tensors it
   passes, `dtype_cases` where the entry has `DType` indices (a dtype parameter is
   written as a parameter), and a `label`. Each optional tensor of an implemented entry is
   passed in at least one row and omitted in at least one. The label is part of the case
   id, which keys nightly history, so renaming it breaks that history.
5. **Write `roofline`.** Inline `flops` (and `bytes`, where the traffic is not simply
   each tensor read or written once) over the same indices, or a `func` that computes
   both from the checked call.

To land an interface before its implementation, write `status: spec-only`: the checks
that need code are skipped. `implemented` turns them on.

## Four recurring forms

### A flag chooses a shape

The layout flags of `GemmFwdOp` decide which axis of `a` carries M. A type family in
`types` states both cases, and each input applies it:

```yaml
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
```

### Optional inputs as a switch

The affine transform of `GroupNormFwdOp` is two optional inputs, and no `affine` flag
beside them: one fact is stated in one place. A workload row passes them through `some`,
and the roofline reads presence with `present`:

```yaml
  signature:
    forall: {B: Dim, C: Dim, L: Shape, T: "DType[float32 | float16 | bfloat16]"}
    params:
      num_groups: {type: int}
      eps: {type: float, default: 1.0e-05}
    inputs:
      x: {dtype: T, shape: "[B, C, *L]"}
      weight: {dtype: T, shape: "[C]", optional: true}
      bias: {dtype: T, shape: "[C]", optional: true}
    outputs:
      output: {dtype: T, shape: "[B, C, *L]"}
    shape_rules:
      - "num_groups > 0 and C % num_groups == 0"
      - "B * (C // num_groups) * prod(L) != 1"
  workloads:
    - {B: 8, C: 128, L: [32, 32], num_groups: 32, dtype_cases: [{T: float16}], label: image}
    - {B: 8, C: 128, L: [32, 32], num_groups: 32, some: [weight, bias],
       dtype_cases: [{T: float16}], label: image-affine}
  roofline:
    flops: "(5 + (1 if present(weight) else 0) + (1 if present(bias) else 0)) * B * C * prod(L)"
```

(Excerpt: the entry has more rows and dtypes.) An op may dispatch on whether an optional
input was passed; it may not read the tensor's contents to decide.

### An input the op writes

`state` in `SSDDecodeFwdOp` is written in place by each decode step, so it declares
`mutated: true` and stays an input; the call returns only `y_out`. The generated operator
names exactly the inputs marked `mutated` as the ones it writes.

```yaml
    inputs:
      A: {dtype: float32, shape: "[H, P, N]"}
      dt: {dtype: float32, shape: "[B, H, P]"}
      x: {dtype: T, shape: "[B, H, P]"}
      B_in: {dtype: T, shape: "[B, G, N]"}
      C_in: {dtype: T, shape: "[B, G, N]"}
      state: {dtype: float32, shape: "[B, H, P, N]", mutated: true, contiguous: true}
    outputs:
      y_out: {dtype: float32, shape: "[B, H, P]"}
```

An output the caller may supply declares `buffer: out` instead: `forward` then takes an
`out` argument, writes it, and returns it.

### A constraint on a metadata tensor's contents

A tensor of offsets or lengths gets its values from a generator in `values`, and states
what its contents must satisfy in `requires`. From `GroupedQueryAttentionVarlenFwdOp`,
where `q_lens` is a `Seq[Int]` index and `T_q` the total query length:

```yaml
cu_seqlens_q: {dtype: int32, shape: "[B + 1]", values: "prefix_sum(q_lens)",
               requires: ["prefix_offsets(T_q)"]}
```

`B` is solved from the generated tensor, so a row gives `q_lens`, not `B`.

## Rules at a glance

**The signature**

- **Order is position.** `params` order is `__init__` order, `inputs` order is `forward`
  order, `outputs` order is return order; reordering is a breaking change.
- **Write against the reference.** Dtypes and parameters follow the authoritative
  reference, never the current code; code that disagrees is fixed, with the entry
  `spec-only` until it conforms.
- **A shared name is an equality.** Tensors of one shape write one shape term; a
  relationship that is not an equality of names is a refinement or a `let`.

**Rules and presence**

- **Refinements read indices, never tensors.** No `x.shape`, no `x is None`, no
  `isinstance`; presence is `present(x)`.
- **Presence is a switch, contents are not.** An op may select its implementation from
  parameters and tensor presence; tensor contents are computation input only.
- **Both sides of an optional input get measured.** Passed and omitted each need a row,
  counted per input.

**Outputs**

- **Output arity is fixed.** One entry has one set of outputs on every call; an op whose
  return changes with a switch is two entries.
- **A written input stays an input.** It declares `mutated: true` and is not listed in
  `outputs`.

## The spec validator {#spec-validator}

Validation is
[`scripts/validate_manifest.py`](https://github.com/tile-ai/TileOPs/blob/main/scripts/validate_manifest.py),
and a spec can be run through it the moment it is written:

```bash
python scripts/validate_manifest.py                           # every entry
python scripts/validate_manifest.py --check-op RMSNormFwdOp   # one entry
python scripts/validate_manifest.py --levels schema,signature # skip the benchmark scan
```

| Level | Checks |
| --- | --- |
| `schema` | top-level fields, `family`, `ref_api`, `composition`, `roofline`, and `types.yaml` |
| `signature` | the signature on every combination of its discriminants, every workload row instantiated, and effects; for an implemented entry, the class's `__init__`, `forward` and declared kernels and sub-ops |
| `bench` | every benchmark takes its calls from the manifest and its roofline from the op |

Checks that read code are skipped for a `spec-only` entry. Kernel selection,
multi-kernel ordering, accumulator dtypes, workspaces, tile sizes and autotuning
configuration are not in the manifest, so the validator does not see them.
