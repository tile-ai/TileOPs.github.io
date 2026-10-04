# Extensions

This page describes forms that only some specs use. Each section covers one case, and each section can be read on its own. The fields most specs need are in [Spec fields](writing.md).

## 1. Optional inputs, nullable outputs and omittable parameters {#presence}

Whether a tensor exists is expressed with `present`. FusedMoESharedExpert is an example:

```yaml
inputs:
  correction_bias:  {dtype: float32, shape: "[E]", optional: true}
  shared_w_gate_up: {dtype: D, shape: "[2 * S, H]", optional: true}
  shared_w_down:    {dtype: D, shape: "[H, S]", optional: "present(shared_w_gate_up)"}
outputs:
  shared_output: {dtype: D, shape: "[T, H]", nullable: "present(shared_w_gate_up)"}
  routed_output: {dtype: D, shape: "[T, H]"}
```

- An optional input is declared with `optional: true`. If several tensors must be passed together or omitted together, they share one discriminant, written `optional: "<expression>"`, as `shared_w_down` does above.
- An output that may be `None` is declared with `nullable: "<expression>"`. In the example above, the op returns `shared_output` when `shared_w_gate_up` is passed, and that output is `None` otherwise.
- The expressions of `optional` and `nullable` may consist only of boolean quantities with finitely many values: `Bool` and enum parameters, ADT tags and fields with finitely many values, and `present(...)`.
- Optional inputs come after required inputs, and `forward` receives them in declaration order with a default of `None`. Omitting an optional input has the same effect as passing `None` explicitly.
- The kind of an `int | None` parameter is `Maybe[Int]`: `present(v)` says whether it is given, and `v.value` is its value. `v.value` may appear only in branches where `present(v)` is true.
- An index is used only on the branches where it appears. In the example above, when `shared_w_gate_up` is not passed, `S` is not used, and the corresponding workload row does not give `S`.
- A workload row lists the optional tensors it passes in `some`. `some` lists only tensors declared `optional: true`; a tensor whose `optional` is an expression follows the value of that expression and is not written in `some`. For example, passing the shared expert is written `some: [shared_w_gate_up]`, and `shared_w_down` is passed along with it.
- For an `implemented` op, each optional tensor must be passed by at least one row and omitted by at least one row.

## 2. Shapes that vary with parameters: type family {#type-family}

When a parameter determines the rank of a tensor or the order of its axes, the shape is described with a type family. A type family is defined in the spec's `signature.types`:

```yaml
signature:
  types:
    Mat:
      params: {t: Bool, R: Dim, C: Dim}
      match: t
      cases:
      - {when: false, is: "[R, C]"}
      - {when: true, is: "[C, R]"}
  inputs:
    a: {dtype: T, shape: "Mat[trans_a, M, K]"}   # arguments map to Mat's t, R, C in order
```

- A tensor's `shape` is written `<Family>[argument, ...]`, and the arguments map to `params` in declaration order.
- The subject of `match` must be a discriminant with finitely many values: `Bool`, an enum, an ADT, `present(...)`, or a tuple of these. When matching a tuple, `when` is written as a list, for example `when: [true, false]`.
- `cases` must have no gaps and no overlaps over all values the spec accepts.
- When several tensors apply the same type family, they always take the same branch.
- References between type families must not form a cycle, and the validator reports an error if a type family is not used by any shape.

A refinement that reads only discriminants is called a **domain restriction**. A domain restriction is checked before a type family branch is selected, and values it excludes need no corresponding case. Clamp is an example:

```yaml
types:
  ClampOut:
    params: {A: Shape, L: Shape, U: Shape, pl: Bool, pu: Bool}
    match: [pl, pu]
    cases:
    - {when: [true, true],  is: "[*broadcast(A, L, U)]"}
    - {when: [true, false], is: "[*broadcast(A, L)]"}
    - {when: [false, true], is: "[*broadcast(A, U)]"}
outputs:
  output: {dtype: T, shape: "ClampOut[A, L, U, present(min), present(max)]"}
shape_rules:
- "present(min) or present(max)"
```

`present(min) or present(max)` reads only discriminants and excludes the case where neither `min` nor `max` is passed, so the three cases already cover every combination Clamp accepts. Whether a refinement reads only discriminants is decided by all the names that appear in it, independent of the order in which the operands are written.

## 3. Parameters with fields: ADT {#adt}

When a parameter has a finite number of forms and each form carries its own fields, the parameter's type is described with an ADT. ADTs are defined in `spec/types.yaml` and shared by several specs:

```yaml
adts:
  MGroupedLayout:
    sum:
      contiguous:
        python: tileops.ops.moe.contracts.ContiguousLayoutSpec
        fields:
          packing: {type: "'tight' | 'aligned'", python: tileops.ops.moe.contracts.ContiguousPacking}
          metadata_kind: {type: "'physical_psum' | 'per_row'", python: tileops.ops.moe.contracts.ContiguousMetadata}
          alignment: Dim
        invariant: "(packing == 'tight') == (alignment == 1) and alignment >= 1"
      masked:
        python: tileops.ops.moe.contracts.MaskedLayoutSpec
        fields: {max_m: Dim}
```

- Each constructor corresponds to a Python class, given by `python`. The object's `kind` attribute is the constructor name, each field is an attribute of the same name, and the value of an enum field is the attribute's `.value`.
- An ADT value is written `{constructor: {field: value}}`, and workload rows use the same form.
- `invariant` is an optional refinement on a constructor, checked both at instantiation and at construction.
- A type family can pattern-match on constructors: `{masked: _}` matches any masked value, and `{contiguous: {metadata_kind: per_row}}` also constrains a field with finitely many values. A field specific to a constructor can be read only in a branch that matched that constructor, for example `layout.max_m` in the masked branch.
- ADTs are sealed: constructors and fields are fixed where the ADT is defined. Adding a constructor to a shared ADT is done by editing its definition directly; a spec that does not accept the new constructor can exclude it with a domain restriction, without changing its type families.

## 4. Computed quantities: let and primitive {#let}

When a shape or formula needs a quantity computed from indices, the quantity is defined as a `let`. MaxPool2d is an example:

```yaml
let:
  kH: "per_axis(kernel_size, 0, 2)"
  sH: "per_axis(stride, 0, 2, fallback=kH)"
  H_out: "pool.out(H_in, kH, sH, pH, dH, ceil_mode)"
outputs:
  output: {dtype: T, shape: "[N, C, H_out, W_out]"}
```

- The value of a `let` is computed from the signature: a `let` that can be evaluated at construction is computed at construction, and the others are computed on every call. Workload rows do not give `let` entries.
- A `let` can refer to other `let` entries, as `sH` refers to `kH`, but dependencies between `let` entries must not form a cycle.

A primitive is a built-in function callable in expressions, such as `broadcast`, `reduced`, `per_axis` and `ceil_div`.

- Primitives, generators and the predicates in `requires` are each a fixed set. The full list is in `tileops.manifest.primitives`, and the docstring of each member describes what it computes.
- Members used by only one family carry a family prefix, such as `pool.out` and `moe.capacity`.
- When an argument is outside its domain, the primitive raises an error that points to the declaration that called it.
- Adding a member requires changing `tileops.manifest.primitives`, with tests.
- Every primitive that takes an axis argument handles axes by the same rule: for a zero-dimensional tensor, both `0` and `-1` denote the single scalar axis; otherwise, the axis ranges over `[-rank, rank)`.

## 5. Construction-time tensors, memory layout and device {#placement}

- A tensor passed at construction is declared in `params` with `dtype` and `shape`, and can declare `optional: true`. For example, `rescale_factors` in LongRoPE:

  ```yaml
  params:
    rescale_factors: {dtype: R, shape: "[D // 2]", optional: true}
  ```

- A tensor that must be contiguous in memory declares `contiguous: true`, for example the inputs and outputs of the MoE staged ops. A tensor without this declaration can have any strides.
- A tensor that must be on the CPU declares `device: cpu`, for example `cu_seqlens_cpu` in GatedDeltaNet.
- An op without call-time tensor inputs declares a `device` parameter, for example Alibi. How the call device is determined is in [Calls and validation § 4](calls.md#device).

## 6. Writes to arguments: effect {#effects}

An op without effect declarations only reads its inputs and allocates new tensors for its outputs. If an op writes to an argument, an effect is declared on the corresponding tensor in the signature. Effects determine the read and write counts of the roofline, and the operator schema generated for an op that declares a compile boundary.

**Table 1** Effect declarations

| No. | Declaration | Meaning | Example |
| --- | --- | --- | --- |
| 1 | `buffer: out` on an output | `forward` gets a parameter `out` after all inputs. When the caller passes `out`, the op writes the result into it and returns it; otherwise the op allocates a new tensor. `out` has the same shape and dtype as that output | `output` of MoEGroupedGemm |
| 2 | `mutated: true` on an input | the op may write this input, and its contents before the call take part in the computation | |
| 3 | `mutated: true` and `write_only: true` on an input | a result buffer that must be passed: the op overwrites it, and the result depends only on the other inputs; if the op returns `None`, `outputs` is empty | `output` of FusedMoEExperts |
| 4 | `mutated: "<discriminant expression>"` on an input | the op writes this input only when the expression is true | `mutated: inplace` of activation functions |
| 5 | `alias: <input name>` on an output | when that input is written, this output is that input object itself | `alias: input` of activation functions |

The `inplace` parameter of activation functions is an example:

```yaml
params:
  inplace: {type: bool, default: false, kw_only: true}
inputs:
  input: {dtype: T, shape: "[*S]", mutated: inplace}
outputs:
  output: {dtype: T, shape: "[*S]", alias: input}
```

Effect declarations must also satisfy the following rules, and the validator rejects specs that violate them:

- `write_only: true` can be used only together with `mutated: true`;
- `alias` must point to an input that is written;
- an output that declares `alias` cannot also declare `buffer`;
- at most one output declares `buffer: out`.

For each effect branch, the validator checks that the operator schema, the aliasing and the roofline read and write counts agree.

## 7. Metadata tensors: generator and requires {#generators}

Varlen, paged and similar ops use metadata tensors such as `cu_seqlens` and `block_table`. The type of a metadata tensor is written in the signature, and its values are generated at instantiation by a generator written in the tensor's `values` field. MeanPooling is an example:

```yaml
forall: {B: Dim, S: Dim, H: Dim, D: Dim, NS: Dim, NC: Dim, T: "DType[...]", seq_lens: "Seq[Int]"}
inputs:
  offsets: {dtype: int32, optional: true, shape: "[NS + 1]",
            values: "prefix_sum(seq_lens)", requires: ["prefix_offsets(S)"]}
  indices: {dtype: int32, optional: "present(offsets)", shape: "[NC, 2]",
            values: "chunk_indices(seq_lens, chunk_size)"}
```

The rules for generators:

- The arguments of a generator are given by the workload row, for example the list of lengths `seq_lens`. A name of kind `Seq[Int]` in `forall` can be used only as a generator argument. A `Seq[Int]` in a workload row can be written as a list of integers or as a call to a value primitive that returns a list, for example `seq_lens: "repeat(512, 64)"`.
- At instantiation, the generator's result is unified with the declared shape, which solves the other indices in the shape, so they are not given in the row. In the example above, `NS` and `NC` are solved from the generated `offsets` and `indices` respectively. In an actual call, these indices are likewise solved from the input tensors.
- A generator is either deterministic or uses private random numbers derived from the workload seed, so the same row generates the same values every time.
- The rank of each generator result is fixed, or determined by its shape arguments.
- A generated tensor declares an integer dtype (`int32` or `int64`). When an argument is outside its domain, or the result is outside the range of the declared dtype, the generator raises an error.
- A generator argument can be a primitive that returns a list, for example `as_tensor(balanced_sizes(M, G))` in GroupedGemm.

The rules for `requires`:

- `requires` lists predicates that constrain the contents of a metadata tensor. For example, `prefix_offsets(S)` requires the tensor's first element to be 0, the elements to be non-decreasing, and the last element to be `S`. The contents of the constrained tensor are the predicate's implicit first argument.
- Each predicate reads the constrained tensor at a fixed rank; predicates that give elementwise bounds (such as `in_range`) can apply to tensors of any rank.
- A predicate argument can be another metadata tensor, so that one predicate constrains the relation between two tensors. For example, `batch_offsets` in GroupedGemm declares `requires: ["exclusive_prefix_of(batch_sizes)"]`, which requires it to be the exclusive prefix sum of `batch_sizes`. On every branch where the constrained tensor exists, the tensor used as the argument must also exist.
- The validator checks `requires` at instantiation against the generated values. In an actual call, the caller guarantees these constraints and the op does not check tensor contents; the validator therefore also checks that the predicate is well defined on every branch where the constrained tensor exists.
- A tensor that declares `requires` must also declare `values`.

## 8. dtype combinations and packed dtypes {#dtype-combos}

When several `DType` indices allow only specific combinations, the spec lists every allowed combination in `dtype_combos`. Paged GQA is an example:

```yaml
forall: {..., T: "DType[float16 | bfloat16 | float8_e4m3fn]", KV: "DType[float16 | bfloat16 | float8_e4m3fn]"}
dtype_combos:
- {T: float16, KV: float16}
- {T: bfloat16, KV: bfloat16}
- {T: float16, KV: float8_e4m3fn}
- {T: bfloat16, KV: float8_e4m3fn}
- {T: float8_e4m3fn, KV: float8_e4m3fn}
```

- Each row is a mapping from index to dtype. All rows have the same keys, the keys may include dtype parameters, and the rows are all distinct.
- The dtype values of a call must equal one of the rows exactly.
- Every column of `dtype_combos` must be used on every branch the spec accepts.

Packed dtypes such as fp4 and int4 are stored in a carrier dtype such as `uint8`, and the spec is written in terms of the carrier:

- `dtype` is the carrier dtype, and `shape` is the carrier shape as PyTorch sees it, for example `packed_weight: "[N, K // 2]"` in GemmW4A16;
- the logical dtype is given by a dtype parameter, or fixed in the spec;
- the roofline counts bytes by the carrier.

## 9. Scalar parameters whose value range depends on dtype {#scalar-dtype}

The valid values of some scalar parameters depend on the dtype at call time, for example `alpha` in Add. Such constraints are written as refinements with the primitives `category` and `representable`:

```yaml
params:
  alpha: {type: int | float, default: 1, kw_only: true}
shape_rules:
- "category(alpha) == 'int' or category(alpha) == category(T)"
- "representable(alpha, T)"
```

The generated call checks apply the same rule to TileOPs in-tree implementations and to implementations provided by a target.

## 10. Composite ops: composition {#composition}

A composite op uses `composition` to record the sub-ops its in-tree implementation may hold, and the positions of its own kernels. FusedMoESharedExpert is an example:

```yaml
composition:
  kind: composite
  stages:
  - {name: route_select, op: FusedTopKFwdOp}
  - {name: routed_experts, op: FusedMoEExpertsFwdOp}
  - {name: shared_expert, op: SharedExpertMLPFwdOp, optional: true}
```

- Each stage refers either to an op in the manifest (`op`) or to a key in the op's own `kernel_types` (`kernel`).
- A sub-op that is not held on every call is written as a stage with `optional: true`; whether it is held is decided by code.
- When and how many times sub-ops are constructed, how they are scheduled, and how forward executes are all decided by code.
- The manifest does not prescribe how the roofline of a composite op relates to the rooflines of its stages.
- For an `implemented` op, the validator checks in order that the `op` stages agree with the class's `delegate_types` and the `kernel` stages agree with the class's `kernel_types`.
- `stages` cannot be empty, stage names are all distinct, and `optional` is a boolean.
