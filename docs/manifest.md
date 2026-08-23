# Reading and writing a manifest entry

A conventional operator library is organised around its implementations: kernels
are written and tuned one at a time, and what shapes and dtypes each supports, and
how fast it runs, are described afterwards. TileOPs is organised the other way — an
op's specification is declared first and the implementation is derived from it. That
specification lives in the entries under
[`src/tileops/manifest/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/manifest).

**An entry makes the op an input to the whole system.** Every stage reads the same
declaration rather than reading the implementation:

| Consumer | Reads | Produces |
| --- | --- | --- |
| Code generation | `signature`, `shape_rules` | the op layer's parameter validation, shape inference, and the kernel's call signature |
| Correctness tests | `ref_api`, the dtypes in `workloads` | a numerical comparison against the reference on every workload |
| Performance measurement | `workloads` | the nightly device time on those shapes |
| Roofline | the variables and formulas in `roofline` | the FLOPs and bytes one call moves — the denominator of efficiency |
| The compile contract | `torch_compile_fullgraph` | the gate on whether `fullgraph=True` compiles |
| Documentation | every field | the support matrix, the op list, the API reference |
| [The validator](https://github.com/tile-ai/TileOPs/blob/main/scripts/validate_manifest.py) | every field | five levels of checking that declaration and implementation agree — see [How to write an entry](#how-to-write-an-entry) |

**For an op with no entry, none of those rows happen**: no generated validation, no
comparison, no performance data, and nothing in CI holding a regression back.
**Writing a manifest entry is not documenting the op; it is connecting the op to that
flow.**{ .keystone }

This page has five parts: (1) what an entry is made of; (2) how to read one; (3) how
to write one; (4) the dimensions committed at construction; (5) the rules worth
remembering. Six real entries close it out.

## The manifest first, the implementation after

**The direction is one-way: the manifest is written first, the implementation
follows.** An entry is written against an authoritative reference — usually
PyTorch's public API — never derived from the TileOPs code as it stands. Ops, tests
and benchmarks are generated from it and validated against it. Two things follow:
an implementation can be regenerated from its spec while the reverse cannot, and
where code and documentation disagree, the manifest governs.

**An entry puts the op inside CI's reach.** Every field has a definite consumer:

| Field | Read by | On a mismatch |
| --- | --- | --- |
| `signature.inputs` / `outputs` | the op layer's validation, `_infer_output_shapes`, a backend's `build_kernel` signature | L1 fails: parameter names or order disagree with `forward` |
| `signature.shape_rules` | the op layer's shape validation | L2 fails: not valid Python, or disagrees with `_infer_output_shapes` |
| `signature.dtype`, `dtype_combos` | the op layer's dtype validation | L3 fails: unknown dtype, or disagrees with `_validate_dtypes` |
| `workloads` | the benchmark's shapes and dtypes | L4 fails: the benchmark does not take its shapes from `load_workloads` |
| `roofline` | `op.eval_roofline()`, the efficiency column in a performance report | a variable that cannot be bound is an error |
| `torch_compile_fullgraph` | the compile-contract gate | declared without a registered compile test fails `compile-contract-gate` |
| `source` | the validator, locating the kernel, op, test and benchmark | a path that does not exist is an error |

**None of those checks run for an op with no entry.** Writing a manifest entry is
not documenting the op; it is placing it under automated checking.

## What an entry is made of

One YAML file per family — a large family may shard — with `op name → entry` at
the top level. All files merge at load time, and a duplicate op name is an error.

The key is the op's Python class name: `{Name}{Direction}Op`, where the direction
is `Fwd` or `Bwd` and is required once both directions exist in the manifest. The
validator requires `cls.__name__` to equal the key character for character; it
resolves nothing by heuristic.

| Field | Required | Contents |
| --- | --- | --- |
| `family` | yes | the family, which decides the file the entry lives in |
| `ref_api` | yes | the fully qualified reference API, e.g. `torch.nn.functional.rms_norm`, or `"none"` |
| `status` | yes | `spec-only` or `implemented`, which decides how far validation runs |
| `torch_compile_fullgraph` | no | literal `true` only; omit for no promise — `false` is invalid |
| `signature` | yes | the interface itself, below |
| `workloads` | yes | the shapes and dtypes the benchmark uses |
| `roofline` | yes | the performance model |
| `source` | yes | paths to the kernel, op, test and benchmark, and the kernel-name map |

`signature` has six sub-fields:

| Sub-field | Contents |
| --- | --- |
| `inputs` | tensor inputs, `name → {dtype, shape?, constraints?, optional?, mutated?, layout?}` |
| `outputs` | tensor outputs, same fields, but an output's shape must be fully specified |
| `params` | non-tensor parameters, `name → {type, default?, kw_only?}` |
| `shape_rules` | shape relationships, as strings holding Python expressions |
| `dtype_combos` | cross-tensor dtype combinations; declared only where the supported set is a strict subset of the product |
| `static_dims` | dimensions the user commits to when constructing the op; arbitrary-rank ops only |

**Key order in `inputs`, `outputs` and `params` is signature position**, so
reordering is a breaking change and readers need an order-preserving parser.

## How to read an entry

Four steps, each answering one question.

1. **`inputs` and `outputs`** — which tensors a call passes, what dtypes each
   accepts, and what comes back.
2. **`params`** — the non-tensor parameters. Whether one belongs to construction
   or to the call follows the reference API; the manifest does not encode the
   distinction. A param with a `default` must carry the same default in the op.
3. **`shape_rules`** — everything `shape` cannot say: divisibility, the range a
   `dim` may take, how the output shape is derived.
4. **`workloads`** — the shapes and dtypes this op is measured on. Every row on a
   Benchmarks page comes from here.

Four ways a dtype is written:

| Written | Means |
| --- | --- |
| `float16 \| bfloat16` | one of these |
| `same_as(x)` | the same dtype as `x` at runtime. It speaks only about dtype, never shape, and adds no axis to the combination count |
| `promote_int_to_float(x)` | `float32` when `x` is integral, otherwise `same_as(x)`. Allowed in `outputs` only |
| `dtype_combos` | the supported cross-tensor combinations, listed. Absent means every combination of the declared unions is supported |

For shapes, look first for `shape`:

- **With `shape`** — fixed rank, dimensions named: `"[B, M, K]"`. **A shared name
  means equality**: `K` in two tensors requires those axes to match.
- **Without `shape`** — arbitrary rank, with every constraint in `params` and
  `shape_rules`.

To read entries programmatically:

```python
from tileops.manifest import load_manifest, load_workloads

ops = load_manifest()                      # every entry, merged
entry = ops["RMSNormFwdOp"]
entry["signature"]["inputs"].keys()        # dict_keys(['x', 'weight'])

load_workloads("RMSNormFwdOp")             # that op's workload rows
```

## How to write an entry

Five steps, each one checkable immediately.

1. **Name it and pick the family.** The key is the class name; the entry goes in
   the file its `family` names.
2. **Write `signature`.** Tensors go in `inputs` / `outputs`, everything else in
   `params`, in call order, with optional inputs after the required ones. Declare
   the dtypes the reference API supports, not the ones the current kernel does.
3. **Write `shape_rules`.** An output's shape has to be fully determined by `shape`
   and `shape_rules` together. For ops with a `dim`, use the helpers in
   [`shape_rules.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/manifest/shape_rules.py) — `dim_range_validity`, `reduced_shape` and the rest — which
   the op layer calls too, so the two cannot disagree.
4. **Write `workloads`.** For a single-tensor-input op the shape key must be
   `{input}_shape`, and every other key must be a `params` name or the reserved
   `dtypes` / `label`.
5. **Write `roofline` and `source`.**

Validation is [`scripts/validate_manifest.py`](https://github.com/tile-ai/TileOPs/blob/main/scripts/validate_manifest.py).

To land an interface before its implementation, write `status: spec-only`:
validation then runs L0 only. Switching to `implemented` turns on L0 through L4.

```bash
python scripts/validate_manifest.py                           # every entry
python scripts/validate_manifest.py --check-op SoftmaxFwdOp    # one entry, all five levels
```

The five levels: L0 structure, L1 parameter names and order against `forward`,
L2 `shape_rules` syntax and agreement with shape inference, L3 dtype names and
agreement with dtype validation, L4 whether the benchmark takes its shapes from
`load_workloads`.

## Dimensions committed at construction: `static_dims`

`static_dims` declares the dimension values a user commits to when constructing the
op instance. It is for arbitrary-rank ops only — a fixed-rank op takes its
dimensions from `shape`.

```yaml
static_dims:
  N: "x.shape[dim]"
```

**The expression is a validation rule for call time, not a derivation at
construction.** Two moments share one contract:

| Moment | What happens |
| --- | --- |
| `__init__` | The commitment. The user's value is stored on `self`; the expression is not evaluated — there is no tensor yet |
| `forward` | The check. The expression is evaluated against the actual tensor and must equal the committed value |

Four rules:

- Every key is a **required** `__init__` keyword parameter, with no default. The
  committed value comes from the user at construction.
- The expression must be a **single-axis reference**, `<tensor>.shape[<const or
  param>]`. Multi-axis forms are forbidden: no `product(...)`, no comprehensions,
  no arithmetic over a shape.
- The tensor named must appear in `signature.inputs`; an axis name that is not an
  integer literal must be a `signature.params` name.
- Key order is the order those keywords appear in the generated `__init__`.

The expression may reference any input tensor, not only the first. In
`torch.nn.functional.linear`, `out_features` can only be bound to `weight` — no
expression over `input.shape` is equivalent:

```yaml
LinearFwdOp:
  signature:
    inputs:
      input: {dtype: "float16 | bfloat16"}
      weight: {dtype: "same_as(input)"}
      bias: {dtype: "same_as(input)"}
    outputs:
      output: {dtype: "same_as(input)"}
    static_dims:
      in_features: "input.shape[-1]"
      out_features: "weight.shape[0]"
    shape_rules:
      - "weight.shape == (out_features, in_features)"
      - "bias.shape == (out_features,)"
      - "output.shape == input.shape[:-1] + (out_features,)"
```

**An empty `static_dims` is legal** — typically a reduction that accepts
`dim=None`, where the extent depends on the whole input shape rather than on a
hyperparameter the user gives. The op author then **must override `_cache_key`**:
the default keys on the full input shape, which is correct but recompiles for every
distinct shape under dynamic shapes. The base class warns once when that happens.

## Rules worth remembering

| Rule | Detail |
| --- | --- |
| Order is position | Key order in `inputs`, `outputs` and `params` is signature position; reordering breaks callers |
| Declare the whole interface | `params` covers every parameter of the reference API, even where the kernel supports only the default |
| An output's shape is fully determined | By `shape` and `shape_rules` together. An input may omit `shape` |
| No shape aliasing | Each tensor declares its own shape; relationships go in shared dimension names or `shape_rules` |
| `optional` is for inputs | A tensor input declares `optional: true`; a param expresses optionality with `default` |
| Output arity is fixed | The names and number of outputs are the same on every call. An op whose return changes with a switch is two entries |
| One entry, one memory order | A different memory order is a different entry — it changes what an axis means |
| A param never selects a shape | It may size a dimension; it may not decide which shape a tensor has |
| Presence is a switch, contents are not | An op may read whether an optional input was passed and dispatch on it. It may not read the tensor's contents to dispatch |
| Both sides of an optional input get measured | Passed and not passed each need at least one workload row. Counted per input, not per combination |

## Six entries

All six are real entries, each covering one form: (1) fixed rank; (2) arbitrary
rank; (3) a param deciding the output shape; (4) optional inputs forming one
switch; (5) an input the op writes; (6) an output dtype promoted from the input.

### 1. Fixed rank, with shared names meaning equality

All three tensors in `BmmFwdOp` declare a `shape`. `B` and `K` appear in both
inputs, so those axes must match; `shape_rules` adds the divisibility the kernel
needs.

```yaml
BmmFwdOp:
  ref_api: torch.bmm
  family: gemm
  status: implemented
  signature:
    inputs:
      a: {dtype: "float16 | bfloat16", shape: "[B, M, K]"}
      b: {dtype: "same_as(a)", shape: "[B, K, N]"}
    outputs:
      d: {dtype: "same_as(a)", shape: "[B, M, N]"}
    shape_rules:
      - "a.shape[0] == b.shape[0]"
      - "a.shape[2] == b.shape[1]"
      - "a.shape[2] % 16 == 0"
      - "d.shape == (a.shape[0], a.shape[1], b.shape[2])"
```

### 2. Arbitrary rank, with the constraints in `shape_rules`

`RMSNormFwdOp` puts no limit on the input's rank — the parameter
`normalized_shape` names the axes reduced over — so no tensor declares a `shape`
and every relationship lands in `shape_rules`.

```yaml
  signature:
    inputs:
      x: {dtype: "float16 | bfloat16"}
      weight: {dtype: "same_as(x)"}
    outputs:
      output: {dtype: "same_as(x)"}
    params:
      normalized_shape: {type: "list[int] | tuple[int, ...]"}
      eps: {type: "float | None", default: null}
    shape_rules:
      - "len(normalized_shape) > 0"
      - "tuple(x.shape[-len(normalized_shape):]) == tuple(normalized_shape)"
      - "weight.shape == tuple(normalized_shape)"
      - "output.shape == x.shape"
```

### 3. A param deciding the output shape

The two layout flags on `GemmFwdOp` decide which axis carries M and which carries
N, so the output shape is a rule with a conditional in it rather than a `shape`
declaration.

```yaml
  signature:
    inputs:
      a: {dtype: "float16 | bfloat16"}
      b: {dtype: "same_as(a)"}
    outputs:
      d: {dtype: "same_as(a)"}
    params:
      trans_a: {type: bool, default: false}
      trans_b: {type: bool, default: true}
    shape_rules:
      - "d.shape == ((a.shape[1] if trans_a else a.shape[0]), (b.shape[0] if trans_b else b.shape[1]))"
```

### 4. Optional tensor inputs forming one switch

The affine transform on `GroupNormFwdOp` is expressed by two optional inputs. They
are two halves of one switch, and that relationship goes in `shape_rules`, treated
no differently from a shape constraint. A rule that reads an optional input writes
its own guard rather than relying on the rule "not applying" when the input is
absent.

```yaml
  signature:
    inputs:
      x: {dtype: "float32 | float16 | bfloat16"}
      weight: {dtype: "same_as(x)", optional: true}
      bias: {dtype: "same_as(x)", optional: true}
    outputs:
      output: {dtype: "same_as(x)"}
    params:
      num_groups: {type: int}
      eps: {type: float, default: 1.0e-05}
    shape_rules:
      - "(weight is None) == (bias is None)"
      - "weight is None or weight.shape == (x.shape[1],)"
      - "bias is None or bias.shape == (x.shape[1],)"
      - "x.shape[1] % num_groups == 0"
      - "output.shape == x.shape"
```

There is no `affine: bool` param: the tensors' presence *is* the switch, and a
param beside it would record the same fact twice, with the two able to disagree.

Both sides need a workload row — the `weight_shape` and `bias_shape` keys present
means passed, absent means not:

```yaml
  workloads:
    - {x_shape: [8, 128, 32, 32], num_groups: 32, dtypes: [float16, bfloat16], label: "image-g32"}
    - {x_shape: [8, 128, 32, 32], num_groups: 32, weight_shape: [128], bias_shape: [128],
       dtypes: [float16, bfloat16], label: "image-g32-affine"}
```

### 5. An input the op writes

A tensor input the op may write declares `mutated: true`. `state` in
`SSDDecodeFwdOp` is one: a decode step writes the new state back into that tensor,
and the call returns only `y_out`.

```yaml
SSDDecodeFwdOp:
  ref_api: "none"
  family: mamba
  status: implemented
  signature:
    inputs:
      A: {dtype: "float32", shape: "[H, P, N]"}
      dt: {dtype: "float32", shape: "[B, H, P]"}
      x: {dtype: "float16 | bfloat16", shape: "[B, H, P]"}
      B_in: {dtype: "same_as(x)", shape: "[B, G, N]"}
      C_in: {dtype: "same_as(x)", shape: "[B, G, N]"}
      state: {dtype: "float32", shape: "[B, H, P, N]", mutated: true}
    outputs:
      y_out: {dtype: "float32", shape: "[B, H, P]"}
    params: {}
    shape_rules:
      - "H % G == 0"
```

Three things follow:

- **A written input stays an input.** Output arity does not change and the return
  does not alias it — `state` is not in `outputs`.
- The declaration matches the operator boundary the op registers: the inputs its
  `mutates_args` names are exactly the ones marked `mutated: true`.
- If contiguity normalisation had to copy that tensor, the op writes the result
  back after the launch.

### 6. An output dtype promoted from the input

`torch.reciprocal` accepts integral inputs and returns `float32`, while floating
inputs round-trip. That promotion is written as `promote_int_to_float`; the op
layer mirrors it, and the validator expands it to a concrete set before checking
`_validate_dtypes`.

```yaml
ReciprocalFwdOp:
  ref_api: "torch.reciprocal"
  signature:
    inputs:
      input: {dtype: "float16 | bfloat16 | float32 | int8 | int16 | int32 | int64 | uint8"}
    outputs:
      output: {dtype: "promote_int_to_float(input)"}
```

## What the validator does not do

Manifest validation checks syntax and consistency; it does not evaluate:

- `shape_rules` are checked for syntax only. They are not evaluated, no call
  pattern is enumerated, and a rule about presence is not distinguished from a
  rule about shape.
- A wrong call is caught by the op's own runtime checks. Passing `weight` without
  `bias` passes manifest validation; the error comes from [`GroupNormFwdOp.forward`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/ops/norm/group_norm.py).
- The manifest does not describe kernel internals: multi-kernel ordering,
  accumulator dtypes, persistent state, tile sizes and autotuning config are all
  outside it.

The full field specification and every rule are in [Op Manifest](design/manifest.md).
