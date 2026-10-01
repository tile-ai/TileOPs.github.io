# Spec fields

Most specs use only the following fields, and this page describes how to write them:

1. `forall`;
1. the `shape` and `dtype` of tensors;
1. construction parameters;
1. simple `shape_rules`;
1. workload rows;
1. inline roofline formulas.

How to write optional inputs, shapes that vary with parameters, side effects, metadata tensors and similar cases is in [Extensions](extensions.md).

## 1. Structure of a spec {#anatomy}

The SiluAndMul spec uses only the fields above:

```yaml
SiluAndMulFwdOp:
  family: elementwise
  status: implemented
  signature:
    forall: {M: Dim, N: Dim, T: "DType[float16 | bfloat16 | float32]"}
    inputs:
      x: {dtype: T, shape: "[M, 2 * N]"}
    outputs:
      output: {dtype: T, shape: "[M, N]"}
  workloads:
  - {M: 2048, N: 14336, dtype_cases: [{T: float16}, {T: bfloat16}], label: llama-8b-ffn-prefill}
  - {M: 1, N: 14336, dtype_cases: [{T: bfloat16}], label: llama-8b-ffn-decode}
  roofline:
    flops: "6 * M * N"
```

The following lists every top-level field a spec can contain. Each comment gives the section that describes the field:

```yaml
<Op>:
  family: <module name>              # 2
  status: implemented | spec-only    # 2
  ref_api: <fully qualified name>    # 2, optional
  signature:
    forall: {<index>: <kind>}        # 3
    params: {<p>: {type, default, kw_only}}                     # 4
    inputs: {<t>: {dtype, shape, optional, mutated, values, requires, ...}}  # 5
    outputs: {<t>: {dtype, shape, nullable, buffer, alias, ...}}   # 5
    types: {<Family>: {params, match, cases}}                   # Extensions 2
    let: {<name>: "<expression>"}                               # Extensions 4
    shape_rules: ["<refinement>"]                               # 6
    dtype_combos: [{<DType index>: <dtype>}]                    # Extensions 8
  workloads: [{<params and indices>, some, dtype_cases, label}] # 8
  roofline: {flops, bytes} | {func}                             # 9
  composition: {kind: composite, stages}                        # Extensions 10
```

## 2. Top-level fields {#top}

`family`, `status`, `signature`, `workloads` and `roofline` are required, and `workloads` contains at least one row.

- `family` is the name of the public module the op belongs to. The op can be imported by class name from `tileops.<family>`, and the `__all__` of that module agrees with the manifest. The name of the YAML file that holds the spec is also determined by `family`, see [Reading and writing the manifest § 6](index.md#layout).
- `status` takes one of two values:
  - `implemented` means an implementation that conforms to the spec exists;
  - `spec-only` means no conforming implementation exists yet; the code may be absent or only partly done. Checks that depend on code are skipped only for `spec-only` ops.
- `status` decides only which code-dependent checks run, and does not affect the methods generated from the signature: any class that has a spec gets all the generated methods, including `eval_roofline()`.
- `ref_api` is optional and records the fully qualified name of the API the op follows semantically, for example `torch.matmul`. The validator checks its format, and when the corresponding module can be imported, it also checks that the name exists.

## 3. forall and kind {#forall}

`forall` declares every free type index in the signature and its kind, for example `forall: {M: Dim, N: Dim, K: Dim, T: "DType[float16 | bfloat16]"}`.

**Table 1** Kinds available in `forall`

| No. | kind | Values | How it is determined at call time | How it is written in a workload row |
| --- | --- | --- | --- | --- |
| 1 | `Dim` | a non-negative integer, an axis length | solved by unification of the inputs | an integer |
| 2 | `Shape` | a tuple of `Dim` | solved by unification of the inputs | a list of integers |
| 3 | `DType[a \| b]` | one of the listed dtypes | solved by unification of the inputs | `dtype_cases` |
| 4 | `Seq[Int]` | a list of integers, such as `q_lens` | exists only at instantiation, as an argument to a generator (see [Extensions § 7](extensions.md#generators)) | a list of integers, or a call to a value primitive that returns a list, such as `"repeat(512, 64)"` |

- A `Dim` can be used where an `Int` is required, and a `Shape` can be used where a `Seq[Int]` is required.
- Every axis in a shape is an integer expression. An axis whose kind is not `Dim`, and every element of a sequence expanded with `*p`, must be non-negative; the generated checks confirm this at construction or at call time.
- Values in YAML are converted to Python values by the declared kind or `type`: `Shape` and `tuple[...]` become tuples, `Seq[Int]` and `list[...]` stay lists, dtype names become `torch.dtype`, and ADT values become the corresponding Python objects.
- Within one spec, the names of indices, `let` entries and tensors are all distinct. `out` is a reserved name and cannot be used as the name of a tensor, parameter or index.

## 4. params {#params}

`signature.params` corresponds one to one with the parameter list of `__init__`. An ordinary construction parameter declares `type`, and optionally `default` and `kw_only` (keyword-only). A tensor passed at construction declares `dtype` and `shape` instead, see [Extensions § 5](extensions.md#placement).

- The spec is the authority. For an `implemented` op, the validator compares `params` with `__init__` item by item, and requires the same set of parameters, order, `default` and `kw_only`.
- The only extra parameters the code may have are execution-policy parameters: `kernel_map`, `tune` and `target`, which every op has; implementation objects injected by the caller; and the reserved parameter `config`, which is passed only to the kernel.
- The call-time inputs and output buffers in the signature form, in order, the leading part of the `forward` parameter list. `forward` may append code-defined execution parameters after them; those parameters are not part of the signature.

Construction parameters can appear directly in types without extra annotation:

- an `int` parameter can be written in a shape, for example `num_local_experts` in MoE;
- a `list[int]` or `tuple[int, ...]` parameter can be expanded in a shape as `*p`, for example `normalized_shape` in RMSNorm;
- a dtype parameter (whose `type` is a union of dtype names) can be used directly as a tensor's `dtype`, for example `out_dtype` in Alibi;
- how to write an `int | None` parameter is in [Extensions § 1](extensions.md#presence).

The full mapping from `type` to kind is in the design document [Manifest Table 3](../../design/manifest.md#t-types). Parameters of types such as `float` or an unconstrained `str` do not take part in type inference, but they can appear in `shape_rules` and roofline formulas; the validator checks their `type`, `default`, and how they are used in those expressions.

## 5. inputs and outputs {#tensors}

Each tensor is written `{dtype: ..., shape: "..."}`, which corresponds to the type `Tensor[T, s]`.

- A shape is written `"[" axis, ... "]"`, or as an application of a type family. Each axis can be an expression, `*S` or `*primitive(...)`; `"[]"` denotes a zero-dimensional tensor.
- Tensors with the same shape use the same shape term; for example, the input and output of an elementwise op are both written `[*S]`.
- Each spec has exactly one signature, so every call returns outputs with the same names and the same count. If the number of outputs of an op varies with some switch, or the same parameter can be either a scalar or a tensor, the op is split into several specs.
- An op selects its implementation from its parameters and from which tensors are passed. The contents of a tensor are only input to the computation and do not affect implementation selection.

## 6. shape_rules {#refinement}

Each entry in `shape_rules` is a refinement, a predicate that constrains the values of indices, checked after unification.

```yaml
shape_rules:
- "not is_causal or S_q <= S_kv"
- "D > 0 and D % 2 == 0"
```

- Shapes, `let`, type families, refinements and inline roofline formulas use the same closed expression language, with the same operator precedence as Python. The parts of the expression language are in the design document [Manifest Table 10](../../design/manifest.md#t-lang).
- A refinement can depend only on quantities available at run time. A refinement that can be evaluated at construction is checked at construction; the others are checked on every call.
- Constraints on the contents of a metadata tensor are written in the tensor's `requires`, not in `shape_rules`, see [Extensions § 7](extensions.md#generators).
- A condition narrows kinds within the branch it selects: in the branch where `present(v)` is true, `Maybe[X]` narrows to `X`; in the branch where `x == 'a'` or `x in ('a', 'b')` holds, the values of `x` narrow to those literals. If a comparison uses a literal that is not a member of the corresponding enum or dtype set, the validator rejects the refinement.
- Whether a refinement can be satisfied is the responsibility of the spec's author.
- Forms such as `x.shape == (...)`, `x is None` and `isinstance` are rejected; the corresponding rewrites are in [Calls and validation § 7](calls.md#rejected).

## 7. dtype {#dtype}

A tensor's `dtype` is a dtype expression, which takes one of four forms:

**Table 2** Forms of a dtype expression

| No. | Form | Meaning | Example |
| --- | --- | --- | --- |
| 1 | a `DType` index in `forall` | takes a value in the declared set, solved by unification of the inputs | `a: {dtype: T}` |
| 2 | a dtype parameter | takes the value of the construction parameter | `output: {dtype: out_dtype}` |
| 3 | a constant | a fixed dtype | `cu_seqlens_q: {dtype: int32}` |
| 4 | a dtype primitive | computed from other dtypes | `promote_int_to_float(T)`, `coalesce_dtype(out_dtype, D)` |

If the spec has no `dtype_combos`, each `DType` index takes values in its own set independently. How to write a spec in which several `DType` indices allow only specific combinations is in [Extensions § 8](extensions.md#dtype-combos).

## 8. Workload rows {#workloads}

Each workload row determines one call. The benchmarks, nightly and the manifest tests all generate their calls from workload rows.

```yaml
workloads:
- {M: 2048, N: 14336, dtype_cases: [{T: float16}, {T: bfloat16}], label: llama-8b-ffn-prefill}
```

**Table 3** Keys of a workload row

| No. | Key | Value |
| --- | --- | --- |
| 1 | a construction parameter name | the parameter's value; parameters without a default must be given |
| 2 | `some` | the optional tensors passed in this call, see [Extensions § 1](extensions.md#presence) |
| 3 | a `Dim`, `Shape` or `Seq[Int]` index in `forall` | the indices used on the current branch that are not solved by a generator |
| 4 | `dtype_cases` | a non-empty list, each item of which is one assignment of the `DType` indices used on the current branch, for example `[{T: float16}, {T: bfloat16}]`; written only when such indices exist; dtype parameters are given as construction parameters |
| 5 | `label` | the name of the row |

- An index is used on a branch when it appears in one of the following places on that branch: a shape, a dtype, a refinement, a generator argument, `requires`, or an inline roofline formula. A workload row must give every index used on the current branch, and only those indices; `let` entries and indices solved by a generator are not given in the row.
  - For this decision, each expression is first simplified for the current branch. If the condition of a refinement is always true on that branch, the indices in it do not count as used.
  - When a `let` is used, the indices in its expression also count as used.
  - A roofline in `func` form makes no index used.
  - A discriminant that selects a type family branch, decides whether a tensor exists, or decides whether an output is `None` always counts as used.
- Each row expands into several cases according to `dtype_cases`. The case id joins the following three parts in order with `-`:

  1. `label`;
  1. each dtype value in `dtype_cases`, in the declaration order of `forall`;
  1. the values of dtype parameters, in the declaration order of `params`.

  For example, the id of the first case in the example above is `llama-8b-ffn-prefill-float16`.
- Nightly history is keyed by case id, so changing a `label` breaks the history of that row.
- A `label` is non-empty, at most 24 characters long, and contains only characters in `[A-Za-z0-9._-]`. A `label` describes only the scenario the row models, that is, the model and its use or the synthetic purpose, plus the qualifiers needed to tell similar rows apart, for example `llama-8b-ffn-prefill`. The op name and dtype already appear in the case id and are not repeated. Rows that share a `label` may differ only in dtype.
- Case ids within one spec are all distinct.
- At instantiation, the workload row determines the shapes, dtypes, parameter values, which optional tensors are passed, and the values of metadata tensors. The rest is generated by fixed rules: the device is determined as described in [Calls and validation § 4](calls.md#device), strides are contiguous, tensors do not alias each other, and ordinary data is random.
- The validator re-infers the call from the instantiated inputs and requires the result to agree with the workload row.
- Workload rows are not responsible for unit-test coverage; the shapes needed to cover every branch of the kernel are chosen separately by each op's tests.

## 9. roofline {#roofline}

`roofline` gives the FLOPs and byte count of one call. It is written either as inline formulas or as a `func`:

```yaml
roofline:
  flops: "2 * M * N * K"        # inline formula; when bytes is omitted it is derived from the signature
# or
roofline:
  func: "tileops.perf.formulas.gqa_fwd_roofline"
```

- Inline formulas are written in the expression language and can refer to the signature's indices, construction parameters and `let` entries, as well as `present(t)`, `bytes(t)` (the byte count of tensor `t`) and built-in primitives. When the cost depends on whether a tensor is passed, the formula can distinguish the cases with `present(...)`.
- When `bytes` is omitted, the byte count is derived by reading or writing each tensor in full once:
  - an input that is not written counts as one read, an output counts as one write, a `mutated` input counts as one read and one write, and a `write_only` input counts as one write only;
  - each tensor name in the signature counts as separate storage, even if the caller passes the same tensor to two parameters; only aliases declared in the signature (`buffer`, `alias`) count as the same storage;
  - the byte count of each tensor is `prod(shape) * bits(dtype) / 8`, and packed dtypes are counted by their carrier.
- The byte count is the algorithm's minimum memory traffic: intermediate results are not counted, and an input that the algorithm reads only in part counts the distinct elements actually read. When the two differ by less than 1% on every workload row, the whole tensor may be counted.
- If the derived byte count does not match this minimum traffic, the spec writes `bytes` explicitly, with a corresponding test.
- `flops` is the minimum amount of arithmetic the algorithm needs for the call, not the number of instructions the hardware executes:
  - values already computed count as reused;
  - recurrent ops such as linear attention and state-space scans count by the chunked algorithm at the chunk size when the signature has a chunk size, and by per-token recurrence otherwise;
  - elementwise operations, attention and MoE are counted by the conventions in the design document [Roofline § 1.3](../../design/roofline.md#13-convention);
  - a path that is an identity map on the call's dtype counts as 0.
- `func` is used when the formula needs Python logic. It points to a module-level function `f(call) -> tuple[int, int]` in `tileops.perf.formulas`. The function reads only its argument `call` and does not read the op instance. `call` is one checked call and provides the following interface:

  **Table 4** The argument `call` of a roofline `func`

  | No. | Interface | Contents |
  | --- | --- | --- |
  | 1 | `call.ix` | the parameters, the indices and dtype indices solved for this call, and the `let` entries used, that is, the names an inline formula can refer to |
  | 2 | `call.present(t)` | whether tensor `t` is passed, held or returned; `call.present("out")` says whether the caller passed `out` |
  | 3 | `call.tensors[t]` | the `(shape, dtype name)` of tensor `t` |
  | 4 | `call.bytes(t)` | the byte count of tensor `t` |
  | 5 | `call.values(t)` | the contents of metadata tensor `t`; calling it on meta tensors raises an error, because meta tensors have no values |
  | 6 | `call.stages` | the calls completed by each sub-op of a composite op in this call, indexed by stage name |

- Every spec generates an `eval_roofline()` method, which computes FLOPs and byte count from the op's most recently completed call. The benchmarks obtain the numbers through this method and write them into the results; the roofline tools read the benchmark results and do not call the op directly.

The full roofline rules are in the design document [Roofline](../../design/roofline.md).
