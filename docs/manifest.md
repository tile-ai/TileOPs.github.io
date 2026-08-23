# Reading and writing an op's spec

A conventional operator library is organised around its implementations: kernels are
written and tuned one at a time, and what shapes and dtypes each supports, and how fast it
runs, gets described afterwards.

TileOPs is organised the other way round: an op's specification is declared first, and
the implementation is derived from it. That
specification is the op's **spec**, a YAML declaration under
[`src/tileops/manifest/`](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/manifest); those files together are the manifest.

**A spec makes the op an input to the whole system.** Every stage reads the same
declaration rather than reading the implementation:

| Consumer | Reads from the spec | Produces |
| --- | --- | --- |
| Codegen — the agent writing the op and kernel | `signature`, `shape_rules` | the op layer's parameter validation, shape inference, and the kernel's call signature |
| [pytest](https://github.com/tile-ai/TileOPs/tree/main/tests) | `ref_api`, the dtypes in `workloads` | a numerical comparison against the reference on every workload |
| [The nightly benchmark](https://github.com/tile-ai/TileOPs/tree/main/benchmarks) | `workloads` | the device time on those shapes |
| [Roofline](https://github.com/tile-ai/TileOPs/tree/main/src/tileops/perf) | the variables and formulas in `roofline` | the FLOPs and bytes one call moves — the denominator of efficiency |
| CI's `compile-contract-gate` | `torch_compile_fullgraph` | the test that requires `fullgraph=True` to compile |
| This site | every field | the support matrix, the op list, the API reference |
| CI's [spec validator](https://github.com/tile-ai/TileOPs/blob/main/scripts/validate_manifest.py) | every field | five levels of checking that declaration and implementation agree — see [Writing a spec](#writing-a-spec) |

**Every row presupposes a spec**: without one there is no generated validation, no
numerical comparison, no performance data, and nothing in CI holding a regression back.
**Writing a spec is not documenting the op; it is connecting the op to that flow.**{ .keystone }

The page runs: what a spec contains, how to read one, how to write one, `static_dims`
(the dimensions committed at construction), and optional inputs (where writing a spec goes
wrong most often). Then six real specs as case studies, the rules at a glance to check a
spec against before submitting it, and what the validator does and does not check.

## Codegen and CI

A spec puts the op inside CI's reach. The ops, tests and benchmarks TileOPs ships are
generated from their specs, three places each reading their own part:

- **The op layer's** parameter validation and shape inference follow `signature`.
- **A test** takes its comparison target and dtypes from `ref_api` and `workloads`.
- **A benchmark** reads its shapes through `load_workloads`.

The spec validator then checks that generated code against the spec, level by level, and
a declaration the implementation does not match is an error; which field surfaces how is in
[the spec validator](#spec-validator).

**An op with no spec is bound by none of these checks**, so a new op starts with its
spec: once that lands, CI holds every later change against it.

## What a spec contains

One YAML file per family — a large family may shard — with `op name → spec` at the top
level. The files merge at load time, and a duplicate op name is an error.

The key is the op's Python class name: `{Name}{Direction}Op`, where the direction
is `Fwd` or `Bwd` and is required once both directions exist in the manifest. The
spec validator requires `cls.__name__` to equal the key character for character; it
resolves nothing by heuristic.

| Field | Required | Contents |
| --- | --- | --- |
| `family` | yes | the family, which decides the file the spec lives in |
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

## Reading a spec

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

- `float16 | bfloat16` — one of these.
- `same_as(x)` — the same dtype as `x` at runtime. Dtype only, never shape, and it adds
  no axis to the combination count.
- `promote_int_to_float(x)` — `float32` when `x` is integral, otherwise `same_as(x)`.
  Allowed in `outputs` only.
- `dtype_combos` — the supported cross-tensor combinations, listed one by one. Absent
  means every combination of the declared unions works.

For shapes, look first for `shape`:

- **With `shape`** — fixed rank, dimensions named: `"[B, M, K]"`. **A shared name means
  equality**: `K` in two tensors requires those axes to match.
- **Without `shape`** — arbitrary rank, with every constraint in `params` and
  `shape_rules`.

Reading specs programmatically:

```python
from tileops.manifest import load_manifest, load_workloads

ops = load_manifest()                      # every spec, merged
spec = ops["RMSNormFwdOp"]
spec["signature"]["inputs"].keys()         # dict_keys(['x', 'weight'])

load_workloads("RMSNormFwdOp")             # that op's workload rows
```

## Writing a spec

Five steps, each one checkable immediately.

1. **Name it and pick the family.** The key is the class name, and the spec goes in the
   file its `family` names.
2. **Write `signature`.** Tensors in `inputs` / `outputs`, everything else in `params`,
   in call order, optional inputs after the required ones. Declare the dtypes the
   reference API supports, not the ones the current kernel does.
3. **Write `shape_rules`.** `shape` and `shape_rules` together have to determine an
   output's shape completely. For ops with a `dim`, use the helpers in
   [`shape_rules.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/manifest/shape_rules.py) — `dim_range_validity`, `reduced_shape` and the rest — which the
   op layer calls too, so the two cannot disagree.
4. **Write `workloads`.** For a single-tensor-input op the shape key must be
   `{input}_shape`, and every other key must be a `params` name or the reserved
   `dtypes` / `label`.
5. **Write `roofline` and `source`.**

To land an interface before its implementation, write `status: spec-only` and only L0
runs; `implemented` turns on all five levels. What each level checks, and what the
validator cannot see, are in the last section: [the spec validator](#spec-validator).

## `static_dims`

`static_dims` declares the dimension values a user commits to when constructing the op
instance, and is for arbitrary-rank ops only — a fixed-rank op takes its dimensions from
`shape`.

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

- Every key is a **required** `__init__` keyword parameter, with no default: the
  committed value comes from the user at construction.
- The expression must be a **single-axis reference**, `<tensor>.shape[<const or
  param>]`. Multi-axis forms are forbidden: no `product(...)`, no comprehensions,
  no arithmetic over a shape.
- The tensor named must appear in `signature.inputs`; an axis name that is not an
  integer literal must be a `signature.params` name.
- Key order is the order those keywords appear in the generated `__init__`.

Single-axis is the rule most often broken, and both rejected forms fail for the same
reason: neither can be checked one axis at a time in `forward`.

```yaml
static_dims:
  in_features: "input.shape[-1]"        # accepted
  out_features: "weight.shape[0]"       # accepted: any input tensor may be referenced
# numel: "product(x.shape)"             # rejected: multi-axis
# last: "x.shape[x.ndim - 1] * 2"       # rejected: arithmetic over a shape
```

Which tensor is referenced is unrestricted. In `torch.nn.functional.linear`,
`out_features` can only be bound to `weight`; no expression over `input.shape` is
equivalent:

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

**An empty `static_dims` is legal.** The typical case is a reduction that accepts
`dim=None`: the extent depends on the whole input shape rather than on a hyperparameter
the user gives, so there is nothing to commit at construction.

When it is empty, the op author **must override `_cache_key`**. The default keys on the
full input shape — correct, but under dynamic shapes every new shape recompiles; the base
class warns once when it sees this.

```python
class SumFwdOp(Op):
    def _cache_key(self, x_shape):
        return (math.prod(x_shape),)   # full reduce: equal numel shares a kernel
```

## Optional inputs {#optional-inputs}

An input that may be absent pulls `shape_rules`, `workloads`, `roofline` and the dtype
declarations along with it, which is why this is where writing a spec goes wrong most
often. Each of the ten questions below answers one decision, every snippet comes from a
real spec in the repository, and they run in this order:

- **1–2**: what an optional input is, and when a call counts as having passed one.
- **3–4**: the two decisions when writing the spec — whether to dispatch on presence,
  and whether to add a `bool` beside it.
- **5–6**: how `shape_rules` follows.
- **7–9**: how `workloads`, `roofline` and the dtype declarations follow.
- **10**: why output arity does not change with the switch.

**1. How do optional inputs differ from optional params?**

Different fields, because optional means different things for the two:

- **A tensor input** uses `optional: true`. A tensor is either there or not, with
  nothing in between.
- **A param** uses `default`. A param always has a value and falls back to that
  default when the caller gives none.

Two boundaries go with it:

- `optional` is allowed under `inputs` only. **An output cannot be optional** — the
  caller has to know in advance how many values it will get back.
- An output buffer the caller prepares for the op to write into (`out=`) is a param,
  not an optional input.

```yaml
# MoePermuteNopadFwdOp
inputs:
  hidden_states: {dtype: "float16 | bfloat16"}
  topk_ids: {dtype: "int32"}
  expert_map: {dtype: "int32", optional: true}   # a tensor: optional
params:
  num_experts: {type: int}
  num_experts_local: {type: int}                 # a param would use default
```

**2. When does a call count as having passed an optional input, and where does it sit
in the parameter list?**

Take `Mamba2FwdOp`. Its spec declares five required inputs and two optional ones:

```yaml
inputs:
  x: {dtype: "float16 | bfloat16", shape: "[B, S, H, P]"}
  dt: {dtype: "float32", shape: "[B, S, H]"}
  A: {dtype: "float32", shape: "[H]"}
  B: {dtype: "same_as(x)", shape: "[B, S, G, N]"}
  C: {dtype: "same_as(x)", shape: "[B, S, G, N]"}
  dt_bias: {dtype: "float32", shape: "[H]", optional: true}
  initial_states: {dtype: "float32", shape: "[B, H, P, N]", optional: true}
```

`forward`'s parameters follow that declaration, and the two optional inputs default to
`None`. So a call has four spellings:

```python
op = Mamba2FwdOp()

y, final = op(x, dt, A, B, C)                        # neither dt_bias nor initial_states
y, final = op(x, dt, A, B, C, dt_bias=None)          # identical to the line above
y, final = op(x, dt, A, B, C, dt_bias=bias)          # dt_bias passed
y, final = op(x, dt, A, B, C, initial_states=state)  # initial_states passed
```

Four spellings, two states. Three things read straight off that snippet:

- **The test is the value after binding** — `None` means not passed.
- **The first two lines land in the same place.** Omitting the keyword and passing
  `None` are not two states, because absence is not a value in the domain; `None` is
  only how Python spells it.
- **What the op reads is that same value.** Where `dt_bias is None` holds, the call has
  no `dt_bias`.

The default is also what fixes the position: **optional inputs come after the required
ones.** `forward` receives its arguments in `inputs` order, and a parameter with a
default cannot precede one without. Where the reference API puts an optional argument
in the middle, the spec reorders rather than copying that position.

**3. May an op dispatch on whether an optional input was passed?**

Yes; that is one of the reasons optional inputs exist. `expert_map` on
`MoePermuteNopadFwdOp` is such a switch. Its spec declares it as:

```yaml
# Expert parallelism: supplied when this rank owns num_experts_local of the
# num_experts global experts, absent when it owns them all. Presence picks
# the scan kernel; the ids inside are read only at launch.
expert_map: {dtype: "int32", optional: true}
```

In the op's `forward`, the dispatch reads nothing but whether the argument is there:

```python
# tileops/ops/moe/routed_expert/permute_nopad.py
self.used_expert_map = expert_map is not None

# A map-free kernel and a map-reading one are two different scans, so the
# argument's presence — not the local expert count — picks between them.
kernel = self._get_kernel(
    (hidden_states, topk_ids)
    if expert_map is None
    else (hidden_states, topk_ids, expert_map),
    ...
)
if expert_map is None:
    return kernel(hidden_states, topk_ids)
return kernel(hidden_states, topk_ids, expert_map)
```

That code tests `expert_map is None` and never a value inside `expert_map`, and that
is as far as an optional input goes: **its presence may pick the kernel, its contents
may not.** The shape and the presence are declared in the spec, the contents are not,
and the ids in `expert_map` are read only once the kernel has launched.

**4. A feature that can be switched off — an optional `inputs` entry, or a `bool` param?**

An optional `inputs` entry, with no `bool` alongside it.

Take GroupNorm's affine. The reference API spells it as an `affine=True` switch plus a
pair of weights, and copying that over gives this spec:

```yaml
# not recommended: affine and the two optional tensors record the same fact
inputs:
  x: {dtype: "float32 | float16 | bfloat16"}
  weight: {dtype: "same_as(x)", optional: true}
  bias: {dtype: "same_as(x)", optional: true}
params:
  num_groups: {type: int}
  eps: {type: float, default: 1.0e-05}
  affine: {type: bool, default: true}
```

Whether this call does affine is a single fact, yet this spec records it twice: once in
the value of `affine`, and once in whether `weight` was passed. Kept in two places, the
two can drift apart, and spec validation accepts either way they do:

- `affine: true` with no `weight` — the op is asked for affine and has no weights.
- `affine: false` with a `weight` — a tensor arrives that nothing will read.

The recommended spec keeps a single source and never names `affine`:

```yaml
# GroupNormFwdOp's spec
inputs:
  x: {dtype: "float32 | float16 | bfloat16"}
  weight: {dtype: "same_as(x)", optional: true}
  bias: {dtype: "same_as(x)", optional: true}
params:
  num_groups: {type: int}
  eps: {type: float, default: 1.0e-05}
```

The op reads `weight is not None` where it would have read `affine`: same behaviour, and
neither disagreement can be written down any more.

The test generalises: **if a `bool` param's value always follows from whether some
optional input is there, do not declare it.**

**5. Where do I state that two optional inputs go together?**

In `shape_rules`, treated no differently from a shape constraint, with no new field:

```yaml
shape_rules:
  - "(weight is None) == (bias is None)"          # two halves of one switch
  - "not (min is None and max is None)"           # at least one of them
  - "running_mean is None or not use_input_stats" # presence tied to a param's value
```

**6. How is a rule that reads an optional input written?**

With its own guard — `X is None or <the test that uses X>`. The guard has to precede
the use, not lead the line.

```yaml
# MoePermuteNopadFwdOp
shape_rules:
  - "expert_map is None or expert_map.shape == (num_experts,)"       # accepted
# - "expert_map is not None and expert_map.shape == (num_experts,)"  # not accepted
```

The `and` form is rejected: every `shape_rules` entry is a conjunct that must hold, and
this one evaluates false on a legal call that omits the input, condemning a legal call.

**7. How many workload rows does an optional input need?**

One each for passed and not passed. **Counted per input, not per combination** — n
optional inputs are 2n states, not 2ⁿ. The state is encoded by the key:
`<input>_shape` present means passed, absent means not.

```yaml
# MoePermuteNopadFwdOp: a row on each side of expert_map
workloads:
  - {hidden_states_shape: [1, 7168], topk_ids_shape: [1, 8], num_experts: 384,
     num_experts_local: 384, dtypes: [bfloat16], label: "kimi-k2-decode"}
  - {hidden_states_shape: [1, 7168], topk_ids_shape: [1, 8], num_experts: 256,
     num_experts_local: 128, expert_map_shape: [256], dtypes: [bfloat16],
     label: "deepseek-v3-ep2-decode"}
```

A family whose rows give shapes as scalar dims writes the key anyway: the
`GemmFp8FwdOp` row that passes `bias` carries `bias_shape: [2112]` next to `n: 2112`.

**8. How does roofline account for an optional input?**

In the inline form, presence is a boolean in `vars`, and `flops` and `bytes` reference
that `vars` name — a tensor name does not resolve in the arithmetic layer at all.

```yaml
# GroupNormFwdOp
roofline:
  vars:
    N: "x.shape[0]"
    C: "x.shape[1]"
    spatial_size: "product(x.shape[2:])"
    affine: "weight is not None"                       # presence lives in vars
  flops: "(5 if affine else 3) * N * C * spatial_size" # arithmetic reads the var
```

Where the formula needs the optional input's own shape, switch to
`roofline: {func: ...}`, which reads it from the call.

**9. Can `dtype_combos` or `same_as()` reference an optional input?**

No. Both must hold on every call, and on a call that omits the input the name points
at nothing.

```yaml
# not accepted: weight is optional, and both places below assume it is there
inputs:
  x: {dtype: "float32 | float16", shape: "[B, S, H]"}
  weight: {dtype: "same_as(x)", shape: "[H]", optional: true}
  bias: {dtype: "same_as(weight)", shape: "[H]", optional: true}
dtype_combos:
  - {x: float16, weight: float16}
```

Each place fails for its own reason:

- A key of `dtype_combos` must exist on every call, and a call without `weight` cannot
  make up this combination.
- The ref in `same_as(weight)` is optional, so with `weight` absent there is nothing to
  resolve `bias`'s dtype against.

The fix anchors both to a required input:

```yaml
inputs:
  x: {dtype: "float32 | float16", shape: "[B, S, H]"}
  weight: {dtype: "same_as(x)", shape: "[H]", optional: true}
  bias: {dtype: "same_as(x)", shape: "[H]", optional: true}
dtype_combos:
  - {x: float16}
```

**10. What about an op whose number of return values changes with a switch?**

Two specs — or fix the outputs. One spec's output names and count are the same on
every call, otherwise the caller cannot unpack a return whose shape it does not know.
`Mamba2FwdOp` takes the second route and records the deviation in a comment:

```yaml
# ... one declared deviation: final_states is always returned (fixed output
# arity; the Op protocol has no conditional-output mechanism), so the upstream
# return_final_states switch does not exist here.
outputs:
  y: {dtype: "float32", shape: "[B, S, H, P]"}
  final_states: {dtype: "float32", shape: "[B, H, P, N]"}
```

## Case studies

Six real specs, each covering one form: fixed rank, arbitrary rank, a param deciding the
output shape, optional inputs forming one switch, an input the op writes, and an output
dtype promoted from the input.

### 1. Fixed rank and shared names

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

### 2. Arbitrary rank

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

### 3. A param decides the shape

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

### 4. Optional inputs as a switch

The affine transform on `GroupNormFwdOp` is expressed by two optional inputs. They
are two halves of one switch, and that relationship goes into `shape_rules`, treated no
differently from a shape constraint.

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

This spec is three rules from [Optional inputs](#optional-inputs) in practice: no
`affine` among the params (question 4), each rule that reads an optional input writing
its own guard (question 6), and a workload row on both sides of the switch (question 7).

That last one is the two rows below — `weight_shape` and `bias_shape` present means
passed, absent means not:

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

### 6. A promoted output dtype

`torch.reciprocal` accepts integral inputs and returns `float32`, while floating
inputs round-trip. That promotion is written as `promote_int_to_float`; the op
layer mirrors it, and the spec validator expands it to a concrete set before checking
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

## Rules at a glance

All ten appeared above, grouped here by what each constrains, so a finished spec can be
read against them one by one before the validator in the next section runs.

**The signature**

- **Order is position.** Key order in `inputs`, `outputs` and `params` is the parameter
  position in `forward`, so reordering rewrites every caller's code — a breaking change.
- **Declare the whole interface.** `params` covers every parameter of the reference API,
  even where the kernel supports only the default: a spec describes the op's interface,
  not this implementation's reach.

**Shapes**

- **An output's shape is fully determined.** `shape` and `shape_rules` together fix it
  uniquely. An input may omit `shape`; an output may not.
- **No shape aliasing.** Each tensor declares its own shape, and relationships between
  tensors go in shared dimension names or in `shape_rules`.
- **A param never selects a shape.** It may size a dimension, but it may not decide
  which shape a tensor has — that would have one spec describe two interfaces.

**Optionality and switches**

- **`optional` is for inputs.** A tensor input declares `optional: true`, a param
  expresses optionality with `default`, and the two are not interchanged.
- **Presence is a switch, contents are not.** An op may read whether an optional input
  was passed and dispatch on it; it may not read what is inside the tensor.
- **Both sides of an optional input get measured.** Passed and not passed each need at
  least one workload row, counted per input rather than per combination.

**Outputs and memory order**

- **Output arity is fixed.** The names and number of outputs are the same on every call.
  An op whose return changes with a switch is two specs, or the caller cannot unpack a
  return of unknown length.
- **One spec, one memory order.** A different memory order is a different spec — it
  changes what an axis means.

## The spec validator {#spec-validator}

Validation is [`scripts/validate_manifest.py`](https://github.com/tile-ai/TileOPs/blob/main/scripts/validate_manifest.py), and a spec can be run through it the
moment it is written:

```bash
python scripts/validate_manifest.py                            # every spec
python scripts/validate_manifest.py --check-op SoftmaxFwdOp    # one op, all five levels
```

How far it gets is decided by `status`: `spec-only` runs L0, `implemented` runs all
five.

### How each field fails {#per-field}

Every field has a definite consumer and a definite failure:

| Field | Read by | On a mismatch |
| --- | --- | --- |
| `signature.inputs` / `outputs` | the op layer's validation, `_infer_output_shapes`, a backend's `build_kernel` signature | The spec validator fails at L1: parameter names or order disagree with `forward` |
| `signature.shape_rules` | the op layer's shape validation | L2: not valid Python, or disagrees with `_infer_output_shapes` |
| `signature.dtype`, `dtype_combos` | the op layer's dtype validation | L3: unknown dtype, or disagrees with `_validate_dtypes` |
| `workloads` | the benchmark's shapes and dtypes | L4: the benchmark does not take its shapes from `load_workloads` |
| `roofline` | `op.eval_roofline()`, the efficiency column in a performance report | a variable that cannot be bound fails the benchmark |
| `torch_compile_fullgraph` | CI's `compile-contract-gate` | declared without a registered compile test fails that check |
| `source` | the spec validator, locating the kernel, op, test and benchmark | a path that does not exist is an error |

### What each level checks {#the-five-levels}

| Level | Checks |
| --- | --- |
| L0 | the spec's structure: whether the required fields are there and typed correctly |
| L1 | parameter names and order against `forward` |
| L2 | `shape_rules` syntax, and its agreement with `_infer_output_shapes` |
| L3 | dtype names, and their agreement with `_validate_dtypes` |
| L4 | whether the benchmark takes its shapes from `load_workloads` |

### What the validator does not do {#not-checked}

The five levels check syntax and consistency rather than evaluating anything, which
leaves three things out of reach:

- **`shape_rules` are not evaluated.** The validator only checks that each rule is a
  well-formed Python expression; it enumerates no call pattern, and does not
  distinguish a rule about presence from a rule about shape.
- **A wrong call is caught by the op itself.** Passing `weight` without `bias` clears
  spec validation, and the error comes from the runtime checks in
  [`GroupNormFwdOp.forward`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/ops/norm/group_norm.py).
- **Kernel internals are not in the manifest.** Multi-kernel ordering, accumulator
  dtypes, persistent state, tile sizes and autotuning config are not what a spec
  describes.

The full field specification and every rule are in [Op Manifest](design/manifest.md).
