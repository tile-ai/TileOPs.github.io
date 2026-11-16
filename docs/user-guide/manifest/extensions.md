# Extensions

This page describes forms that only some specs use. Each section covers one case, and each section can be read on its own. The fields most specs need are in [Spec fields](writing.md).

## 1. Argument types: optional inputs, nullable outputs and omittable parameters {#presence}

A call-time input declares its argument `type` and `default`, as a construction parameter does. `type` lists the members the argument may hold. FusedMoESharedExpert is an example:

```yaml
inputs:
  correction_bias:  {type: "Tensor | None", default: null, dtype: float32, shape: "[E]"}
  shared_w_gate_up: {type: "Tensor | None", default: null, dtype: D, shape: "[2 * S, H]"}
  shared_w_down:    {type: "Tensor | None", default: null, dtype: D, shape: "[H, S]"}
outputs:
  shared_output: {dtype: D, shape: "[T, H]", nullable: "present(shared_w_gate_up)"}
  routed_output: {dtype: D, shape: "[T, H]"}
shape_rules:
- "present(shared_w_down) == present(shared_w_gate_up)"
```

**Table 1** Members of an argument type

| No. | Member | Value it takes | Read as, on a branch narrowed to it |
| --- | --- | --- | --- |
| 1 | `Tensor` | a tensor | the declaration's `Tensor[T, s]` |
| 2 | `list[Tensor]` | a list or tuple of tensors | a tensor sequence, see [§ 11](#sequences) |
| 3 | `int` | an `int` that is not a `bool` | `Int`, a call-time scalar |
| 4 | `float` | a `float` | a value, a call-time scalar |
| 5 | `Number` | an `int` or a `float` | a value, a call-time scalar |
| 6 | `bool` | a `bool` | `Bool`, a call-time scalar |
| 7 | `None` | `None` | `None` |

The rules for argument types:

- `type` defaults to `Tensor`, which stands for the declaration's `Tensor[dtype, shape]`. `Number` already holds `int` and `float`, so it is never listed together with them.
- An argument with a `Tensor` or `list[Tensor]` member is a call-time input, declared in `inputs`; every other argument is a construction parameter, declared in `params`. A tensor is never a construction parameter.
- A value takes a member by its Python type: `None` takes `None`; a list or tuple takes `list[Tensor]`; a tensor takes `Tensor`; a `bool` takes only `bool`; an `int` takes `int`, else `Number`; a `float` takes `float`, else `Number`. A value that no member takes is rejected.
- An input without `default` is passed on every call. An input with `default` may be omitted; `forward` takes the `default` as its default, and omitting the input has the same effect as passing that value.
- Among positional inputs, an input without `default` never follows one with `default`. A `kw_only` input follows every positional one, and may be required.
- A required argument that may be `None` declares the `None` member and no `default`, for example `bias: {type: "Tensor | None", dtype: Wb, shape: "[N]"}`.
- `member(x)` is the name of the member `x` holds, such as `'Tensor'` or `'Number'`, and each argument with several members is one discriminant with finitely many values. `present(x)` means `member(x) != 'None'`, whether the caller passed `x` or its `default` applied.
- `member(x) == '<m>'` narrows `x` to that member, `member(x) != '<m>'` excludes it, and `present(x)` excludes `None`. Narrowing composes through `not`, `and`, `or` and conditional expressions.
- The tensor declarations `shape`, `dtype`, `contiguous`, `device`, `values`, `requires` and effects apply only on the branches where the argument holds a tensor member. An index used only by them is not used elsewhere: in the example above, when `shared_w_gate_up` is not passed, `S` is not used, and the corresponding workload row does not give `S`.

Inputs that must be passed together or omitted together each declare `type: "Tensor | None", default: null`, and a refinement that reads only discriminants ties them, as `present(shared_w_down) == present(shared_w_gate_up)` does above. Such a refinement is a domain restriction, see [§ 2](#type-family).

An output that may be `None` is declared with `nullable: "<expression>"`. In the example above, the op returns `shared_output` when `shared_w_gate_up` is passed, and that output is `None` otherwise. The expression may consist only of discriminants with finitely many values: `Bool` and enum parameters, ADT tags and fields with finitely many values, `present(...)` and `member(...)`.

The kind of an `int | None` parameter is `Maybe[Int]`. `present(v)` says whether it is given, and on a branch where `present(v)` holds, `v` itself is its value, for example `R: "rotary_dim if present(rotary_dim) else D"` in a `let`.

**Call-time scalars.** An argument narrowed to a scalar member is a call-time scalar. RoPECosSin's `seqlen_offsets` is one offset for every sequence or a tensor of one offset per sequence:

```yaml
inputs:
  seqlen_offsets: {type: "int | Tensor", default: 0, dtype: O, shape: "[B]"}
shape_rules:
- "member(seqlen_offsets) != 'int' or 0 <= seqlen_offsets <= P - L"
```

- A call-time scalar's value is read only in refinements and in `let` entries; a `let` that reads one, directly or through another `let`, is likewise read only in refinements and in such `let` entries.
- Shapes, dtypes, type families, `nullable` and `mutated` conditions, `values`, `requires` and roofline formulas never read a call-time scalar's value, and the validator rejects a spec that does. A value therefore decides no output shape and no branch. Construction parameters are not restricted this way.
- The refinements and `let` entries that read a call-time scalar run on every call. The Op plan is keyed by the member each argument holds and by a scalar's type, never by its value.

**Workload rows.**

- `some` lists the arguments with several members that pass their tensor member in the row. Every input passed is listed, including inputs passed together: passing the shared expert is written `some: [shared_w_gate_up, shared_w_down]`.
- Any other argument with several members takes the row's scalar value or `null`, else its `default`. For example, the Clamp row `{A: [4096, 4096], min: -0.5, max: 0.5, ...}` passes both bounds as numbers. A required argument that may be `None` writes `null` to pass `None`.
- For an `implemented` op, every member of every argument with several members that a domain restriction accepts is taken by at least one row.

**Operator schema.** For an op with a compile boundary, each input becomes one or more slots of the generated `torch.library` operator:

**Table 2** Operator slots of an input

| No. | Member | Slot type | Slot name |
| --- | --- | --- | --- |
| 1 | `Tensor`, the only member | `Tensor` | `<name>` |
| 2 | `list[Tensor]` | `Tensor[]` | `<name>` |
| 3 | `Tensor`, among several members | `Tensor?` | `<name>` |
| 4 | `int` | `SymInt?` | `<name>__int` |
| 5 | `float` | `float?` | `<name>__float` |
| 6 | `Number` | `Scalar?` | `<name>__number` |
| 7 | `bool` | `bool?` | `<name>__bool` |
| 8 | `None` | no slot | — |

- An argument with one member takes one slot under its own name. An argument with several members takes one nullable slot per member other than `None`.
- A call passes at most one slot of an argument, and passes none only when `None` is a member; the operator and its fake function both check this.
- A slot name equal to another argument's name or to an execution parameter's name is rejected.

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
- The subject of `match` must be a discriminant with finitely many values: `Bool`, an enum, an ADT, `present(...)`, `member(...)`, a boolean expression of these such as `member(min) == 'Tensor'`, or a tuple of these. When matching a tuple, `when` is written as a list, for example `when: [true, false]`.
- A `Maybe` discriminant matches by its value; only `_` matches it when it is absent.
- `cases` must have no gaps and no overlaps over all values the spec accepts.
- When several tensors apply the same type family, they always take the same branch.
- References between type families must not form a cycle, and the validator reports an error if a type family is not used by any shape.

Clamp is an example of matching on argument members. `min` and `max` each hold a number, a tensor or `None`, and only a tensor bound takes part in broadcasting the output:

```yaml
types:
  ClampOut:
    params: {A: Shape, L: Shape, U: Shape, tl: Bool, tu: Bool}
    match: [tl, tu]
    cases:
    - {when: [true, true],   is: "[*broadcast(A, L, U)]"}
    - {when: [true, false],  is: "[*broadcast(A, L)]"}
    - {when: [false, true],  is: "[*broadcast(A, U)]"}
    - {when: [false, false], is: "[*A]"}
inputs:
  input: {dtype: T, shape: "[*A]"}
  min: {type: "Number | Tensor | None", default: null, dtype: T, shape: "[*L]"}
  max: {type: "Number | Tensor | None", default: null, dtype: T, shape: "[*U]"}
outputs:
  output: {dtype: T, shape: "ClampOut[A, L, U, member(min) == 'Tensor', member(max) == 'Tensor']"}
```

A refinement that reads only discriminants is called a **domain restriction**. A domain restriction is checked before a type family branch is selected, and values it excludes need no corresponding case. MoEPrePermute is an example:

```yaml
types:
  LayoutMetadata:
    params: {layout: MGroupedLayout, E: Dim, P: Dim}
    match: layout
    cases:
    - {when: {contiguous: {metadata_kind: physical_psum}}, is: "[E]"}
    - {when: {contiguous: {metadata_kind: per_row}}, is: "[P]"}
shape_rules:
- "layout.kind == 'contiguous'"
```

`layout.kind == 'contiguous'` reads only the discriminant `layout` and excludes the masked layout, so the two cases already cover every value MoEPrePermute accepts. Whether a refinement reads only discriminants is decided by all the names that appear in it, independent of the order in which the operands are written. Clamp's `present(min) or present(max)`, which requires at least one bound, is also a domain restriction.

## 3. Parameters with fields: ADT {#adt}

When a parameter has a finite number of forms and each form carries its own fields, the parameter's type is described with an ADT. ADTs are defined in `spec/types.yaml` and shared by several specs:

```yaml
adts:
  MGroupedLayout:
    sum:
      contiguous:
        python: tileops._ops.moe.contracts.ContiguousLayoutSpec
        fields:
          packing: {type: "'tight' | 'aligned'", python: tileops._ops.moe.contracts.ContiguousPacking}
          metadata_kind: {type: "'physical_psum' | 'per_row'", python: tileops._ops.moe.contracts.ContiguousMetadata}
          alignment: Dim
        invariant: "(packing == 'tight') == (alignment == 1) and alignment >= 1"
      masked:
        python: tileops._ops.moe.contracts.MaskedLayoutSpec
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

- The value of a `let` is computed from the signature: a `let` that can be evaluated at construction is computed at construction, and the others are computed at call time. Workload rows do not give `let` entries.
- A `let` can refer to other `let` entries, as `sH` refers to `kH`, but dependencies between `let` entries must not form a cycle.

A primitive is a built-in function callable in expressions, such as `broadcast`, `reduced`, `per_axis` and `ceil_div`.

- Primitives, generators and the predicates in `requires` are each a fixed set. The full list is in `tileops.manifest.primitives`, and the docstring of each member describes what it computes.
- Members used by only one family carry a family prefix, such as `pool.out` and `moe.capacity`.
- When an argument is outside its domain, the primitive raises an error that points to the declaration that called it.
- Adding a member requires changing `tileops.manifest.primitives`, with tests.
- Every primitive that takes an axis argument handles axes by the same rule: for a zero-dimensional tensor, both `0` and `-1` denote the single scalar axis; otherwise, the axis ranges over `[-rank, rank)`.

## 5. Memory layout and device {#placement}

- Every tensor is a call-time input. A list of values outside the type system is a construction parameter read only through its length; for example, LongRoPE's `rescale_factors` is a `list[float] | None` parameter, checked by `not present(rescale_factors) or len(rescale_factors) == D // 2`.
- A tensor that must be contiguous in memory declares `contiguous: true`, for example the inputs and outputs of the MoE staged ops. Calls are supported with contiguous inputs ([API Stability](../../stability.md)); a non-contiguous tensor without this declaration is not promised to work.
- A tensor that must be on the CPU declares `device: cpu`, for example `cu_seqlens_cpu` in `GDNFwdOp`.
- An op without call-time tensor inputs declares a `device` parameter, for example Alibi. How the call device is determined is in [Calls and validation § 4](calls.md#device).

## 6. Writes to arguments: effect {#effects}

An op without effect declarations only reads its inputs and allocates new tensors for its outputs. If an op writes to an argument, an effect is declared on the corresponding tensor in the signature. Effects determine the read and write counts of the roofline, and the operator schema generated for an op with a compile boundary.

**Table 3** Effect declarations

| No. | Declaration | Meaning | Example |
| --- | --- | --- | --- |
| 1 | `buffer: out` on an output | `forward` gets a keyword-only parameter `out`, default `None`, after all inputs that do not declare `kw_only`. When the caller passes `out`, the op writes the result into it and returns it; otherwise the op allocates a new tensor. `out` has the same shape and dtype as that output | `output` of MoEGroupedGEMM |
| 2 | `mutated: true` on an input | the op may write this input, and its contents before the call take part in the computation | |
| 3 | `mutated: true` and `write_only: true` on an input | a result buffer that must be passed: the op overwrites it, and the result depends only on the other inputs; if the op returns `None`, `outputs` is empty | no current op |
| 4 | `mutated: "<discriminant expression>"` on an input | the op writes this input only when the expression is true | `mutated: inplace` of activation functions |
| 5 | `alias: <input name>` on an output | when that input is written, this output is that input object itself | `alias: input` of activation functions |
| 6 | input `out` with `mutated: true`, `write_only: true` and `kw_only: true` | the required output buffer of an op that cannot allocate its output: keyword-only without a default, last among the inputs; an output declares `alias: out` | `out` of PagedKVCacheGather |
| 7 | `overlaps: [<input>, ...]` on a written input or a `buffer: out` output | the written tensor may share bytes with the inputs listed | `output` of MoEExpertMLPFwdOp |

What a call returns follows from these declarations:

- an op given `out` writes into it and returns that object;
- an op with several outputs returns a tuple with one slot per declared output, in declaration order, and an absent `nullable` output is `None` in its slot;
- an op whose only effect is writing its inputs declares no outputs and returns `None`.

A tensor a call writes, `out` or a written input, shares no byte with another call tensor unless it lists that input in `overlaps`; a call that breaks this raises `CheckError` before anything runs. The check compares each tensor's byte span, from its first element to the end of its last, so two interleaved views of one buffer overlap. An entry lists an input only when every implementation finishes reading it before writing the tensor, such as a composite op whose earlier stage alone reads it.

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

- an effect is declared only on an input without a scalar member, and never on a tensor sequence;
- `write_only: true` can be used only together with `mutated: true`;
- `alias` must point to an input that is written;
- an output that declares `alias` cannot also declare `buffer`;
- at most one output declares `buffer: out`;
- `overlaps` sits on a written input or a `buffer: out` output, and names other inputs, each once.

For each effect branch, the validator checks that the operator schema, the aliasing and the roofline read and write counts agree.

## 7. Metadata tensors: generator and requires {#generators}

Varlen, paged and similar ops use metadata tensors such as `cu_seqlens` and `block_table`. The type of a metadata tensor is written in the signature, and its values are generated at instantiation by a generator written in the tensor's `values` field. MeanPooling is an example:

```yaml
forall: {B: Dim, S: Dim, H: Dim, D: Dim, NS: Dim, NC: Dim, T: "DType[...]", seq_lens: "Seq[Int]"}
inputs:
  cu_seqlens: {dtype: int32, type: "Tensor | None", default: null, shape: "[NS + 1]",
               values: "prefix_sum(seq_lens)", requires: ["prefix_offsets(S)"]}
  chunk_indices: {dtype: int32, type: "Tensor | None", default: null, shape: "[NC, 2]",
                  values: "chunk_indices(seq_lens, chunk_size)"}
shape_rules:
- "present(chunk_indices) == present(cu_seqlens)"
```

The rules for generators:

- The arguments of a generator are given by the workload row, for example the list of lengths `seq_lens`. A name of kind `Seq[Int]` in `forall` can be used only as a generator argument. A `Seq[Int]` in a workload row can be written as a list of integers or as a call to a value primitive that returns a list, for example `seq_lens: "repeat(512, 64)"`.
- At instantiation, the generator's result is unified with the declared shape, which solves the other indices in the shape, so they are not given in the row. In the example above, `NS` and `NC` are solved from the generated `cu_seqlens` and `chunk_indices` respectively. In an actual call, these indices are likewise solved from the input tensors.
- A generator is either deterministic or uses private random numbers derived from the workload seed, so the same row generates the same values every time.
- The rank of each generator result is fixed, or determined by its shape arguments.
- A generated tensor declares an integer dtype (`int32` or `int64`). When an argument is outside its domain, or the result is outside the range of the declared dtype, the generator raises an error.
- A generator argument can be a primitive that returns a list, for example `as_tensor(balanced_sizes(M, G))` in GroupedGEMM.

The rules for `requires`:

- `requires` lists predicates that constrain the contents of a metadata tensor. For example, `prefix_offsets(S)` requires the tensor's first element to be 0, the elements to be non-decreasing, and the last element to be `S`. The contents of the constrained tensor are the predicate's implicit first argument.
- Each predicate reads the constrained tensor at a fixed rank; predicates that give elementwise bounds (such as `in_range`) can apply to tensors of any rank.
- A predicate argument can be another metadata tensor, so that one predicate constrains the relation between two tensors. For example, `batch_offsets` in GroupedGEMM declares `requires: ["exclusive_prefix_of(batch_sizes)"]`, which requires it to be the exclusive prefix sum of `batch_sizes`. On every branch where the constrained tensor exists, the tensor used as the argument must also exist.
- The validator checks `requires` at instantiation against the generated values. In an actual call, the caller guarantees these constraints and the op does not check tensor contents by default; the validator therefore also checks that the predicate is well defined on every branch where the constrained tensor exists.
- A tensor that declares `requires` must also declare `values`.
- An op with an input whose `requires` holds `prefix_offsets` declares `validate_inputs: {type: bool, default: false, kw_only: true}` in `params`. When the caller sets it, a check generated from the manifest reads the offsets on the CPU before the implementation runs; it synchronizes the device, so it cannot run inside CUDA Graph capture.

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

## 11. Tensor sequences {#sequences}

An argument that holds a list of tensors declares `type: "list[Tensor]"`. Its element-wise indices have the kinds `Seq[Dim]`, `Seq[Shape]` and `Seq[DType[...]]`. Concat is an example:

```yaml
forall: {S: "Seq[Shape]", D: "Seq[DType[float16 | bfloat16 | float32]]"}
params:
  dim: {type: int, default: 0}
inputs:
  tensors: {type: "list[Tensor]", dtype: D, shape: "[*S]"}
outputs:
  output: {dtype: "promote(D)", shape: "[*concat(S, dim)]"}
shape_rules:
- "len(S) >= 1"
workloads:
- {S: [[2048, 1024], [2048, 4096], [2048, 3000]], dim: 1, dtype_cases: [{D: [bfloat16, bfloat16, bfloat16]}], label: uneven-cols}
```

- In the `shape` and `dtype` of a `list[Tensor]` declaration, every element-wise index stands for the element's item: element `i` has shape `S[i]` and dtype `D[i]`, and `len(S)` there is `len(S[i])`. Every other index is shared by all elements.
- Outside the declaration, an element-wise index is the whole list, readable in outputs, refinements, `let` entries and roofline formulas. A `let` is not expanded into the element.
- `Seq[Dim]` holds one axis length per element, while a `Shape` is one tuple of axes shared by all elements.
- The sequence length is the common length of its element-wise indices, and a `list[Tensor]` declaration reads at least one element-wise index.
- Element-wise indices are solved on every element. A shared index is solved from the other tensors first; only when inference stalls is it solved from element 0, then checked on every element.
- An empty sequence gives its element-wise indices empty lists; a shared index that only the sequence could solve stays unsolved, and the call raises an error that names the declaration.
- A tensor sequence is passed on every call and is only read. It declares no effect, `values`, `requires` or `device`, and is not a member of a union.
- Choosing the call device, the 16-byte alignment copy and the alias and overlap checks visit each element. The derived byte count counts each element once, and `bytes(t)` is the sum over the elements.
- A workload row writes an element-wise index as a list with one item per element. A `Seq[DType]` index is written in `dtype_cases` as a list of dtypes, and its value in the case id joins them with `.`, for example `uneven-cols-bfloat16.bfloat16.bfloat16`.
- A `list[Tensor]` output, `None` elements in a sequence, `list[Tensor]` as a member of a union, and an effect on a sequence are all rejected.
