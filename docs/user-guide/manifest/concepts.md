# Concepts

The core of a spec is its signature. Formally, a signature is a polymorphic function type. This page uses GEMM as the example and introduces, in order, the concepts used to describe a signature. How each field is written is in [Spec fields](writing.md) and [Extensions](extensions.md).

## 1. polymorphic function type {#pft}

The GEMM signature in mathematical form:

```
Mat[t: Bool, R: Dim, C: Dim] = match t { false → [R, C]; true → [C, R] }

gemm(trans_a, trans_b : Bool) : ∀ (M N K : Dim) (T : DType[float16 | bfloat16]).
       Tensor[T, Mat[trans_a, M, K]] → Tensor[T, Mat[trans_b, K, N]] → Tensor[T, [M, N]]
```

The corresponding YAML is in [Examples § 1](examples.md#gemm).

`trans_a` and `trans_b` after the function name are construction parameters; `M`, `N`, `K` and `T` after `∀` are the quantified names. The signature has three layers:

**Table 1** The three layers of a signature

| No. | Layer | Mathematical form | Written in the manifest as |
| --- | --- | --- | --- |
| 1 | tensor type | `Tensor[T, s]`, where `T` is the dtype and `s` is the shape | `{dtype: T, shape: "[M, N]"}` |
| 2 | function type | from the types of the input tensors to the types of the output tensors | `inputs`, `outputs` |
| 3 | quantification | `∀` lists every type index and its kind | `forall` |

A function type with quantification over its type indices is called a polymorphic function type. The runtime checks and `eval_roofline()` are both generated from the signature; for an op that declares a compile boundary, the `torch.library` operator and the fake/meta functions are also generated from it. The validator runs its static checks on the signature.

## 2. type index and kind {#index}

A type index is a parameter name in a type, such as `T`, `M`, `N` and `K` in GEMM. Every type index has a kind, and the kind fixes its range of values:

**Table 2** Common kinds

| No. | kind | Values | Example |
| --- | --- | --- | --- |
| 1 | `Dim` | a non-negative integer, an axis length | `M`, `N`, `K` in GEMM |
| 2 | `Shape` | a tuple of `Dim`, used for tensors whose rank is not fixed | the shape `[*S]` of an elementwise op |
| 3 | `DType[...]` | one of the listed dtypes | `T` in GEMM |
| 4 | `Bool`, enum, ADT | finitely many values | `trans_a`, `trans_b` in GEMM |

Every type index in a signature has exactly one source, which is one of the following three:

- declared in `forall`, such as `M` and `T`;
- a construction parameter, whose kind is determined by the parameter's `type`; for example, the `type` of `trans_a` is `bool`, and the corresponding kind is `Bool`;
- defined by `let`, see [§ 5](#let).

## 3. type family {#family}

A type family is a function that gives a shape from the value of a discriminant. A discriminant is a quantity with finitely many values, such as a `Bool` parameter, an enum parameter, an ADT parameter, or whether a tensor is passed.

`Mat` in GEMM is a type family: when `t` is false the result is `[R, C]`, and when it is true the result is `[C, R]`. The shape of tensor `a` is written `Mat[trans_a, M, K]`, so:

- at construction the value of `trans_a` is known, `Mat` selects a branch, and the shape of `a` becomes `[M, K]` or `[K, M]`;
- at call time, the values of `M` and `K` are obtained from the actual shape of `a` through unification.

A type family is defined in the `signature.types` of the spec that uses it.

## 4. ADT {#adt}

An ADT (algebraic data type) is a tagged sum type: an ADT value belongs to one of several constructors, and each constructor has its own fields.

The layout parameter of the MoE staged ops is the ADT `MGroupedLayout`, which has two constructors:

**Table 3** Constructors of `MGroupedLayout`

| No. | constructor | Fields | Meaning |
| --- | --- | --- | --- |
| 1 | `contiguous` | `packing`, `metadata_kind`, `alignment` | the rows of each expert are laid out contiguously, and `packing` decides whether they are aligned to `alignment` |
| 2 | `masked` | `max_m` | each expert occupies a fixed `max_m` rows, and extra rows are marked by a mask |

- A `masked` value has only the field `max_m`, and a `contiguous` value has only the other three fields.
- A type family can pattern-match on the constructor and read `layout.max_m` in the `masked` branch.
- Each constructor corresponds to a Python class; for example, `contiguous` corresponds to `ContiguousLayoutSpec`.
- ADTs shared by several specs are defined in `spec/types.yaml`.

The definition of `MGroupedLayout` and a spec that uses it are in [Examples § 2](examples.md#moe).

## 5. `let` {#let}

`let` names a quantity computed from indices. The output height of MaxPool2d is an example:

```yaml
let:
  kH: "per_axis(kernel_size, 0, 2)"
  H_out: "pool.out(H_in, kH, sH, pH, dH, ceil_mode)"
```

`H_out` is computed from the input height `H_in` and the construction parameters, and is used in the output shape. The value of a `let` is computed from the signature and is not given in workload rows.

## 6. refinement {#refinement}

A refinement is a predicate that constrains the values of indices, written in `shape_rules`. Attention is an example:

- the types `q: [B, S_q, H, D]` and `k: [B, S_kv, H, D]` by themselves do not constrain how `S_q` relates to `S_kv`;
- the refinement `not is_causal or S_q <= S_kv` requires `S_q <= S_kv` in the causal case.

## 7. unification and type inference {#unification}

Unification matches the declared shape against the shape of the actual tensor axis by axis, solves for unknown indices, and checks that each index takes the same value everywhere it appears. Take one GEMM call with `trans_a = trans_b = false`:

```
Declared:  a: Tensor[T, [M, K]]      b: Tensor[T, [K, N]]
Actual:    a: bfloat16, (128, 4096)   b: bfloat16, (4096, 512)
```

**Table 4** Unification for this GEMM call

| No. | Correspondence | Result |
| --- | --- | --- |
| 1 | dtype of `a` against `T` | `T := bfloat16` |
| 2 | axis 1 of `a` against `M` | `M := 128` |
| 3 | axis 2 of `a` against `K` | `K := 4096` |
| 4 | dtype of `b` against `T` | checked equal to `T` |
| 5 | axis 1 of `b` against `K` | checked equal to `K` |
| 6 | axis 2 of `b` against `N` | `N := 512` |

Once every index is solved, the output type `Tensor[T, [M, N]]` is fixed as `bfloat16, (128, 512)`.

Type inference is the process of solving every index at call time; unification is the part of it that handles equalities. Other relations between indices are checked by refinements. Which axis forms unification can solve is in [Calls and validation § 3](calls.md#inference).

## 8. effect {#effect}

An effect describes what a call writes to its arguments and how outputs alias inputs. The `inplace` parameter of an activation function is an example: when it is true, the op writes the input tensor and returns that input object as the output:

```yaml
inputs:  {input: {dtype: T, shape: "[*S]", mutated: inplace}}
outputs: {output: {dtype: T, shape: "[*S]", alias: input}}
```

Effects determine the read and write counts of the roofline, and the operator schema (its `mutates_args`) generated for an op that declares a compile boundary. All effect declarations are in [Extensions § 6](extensions.md#effects).

## 9. Glossary {#glossary}

**Table 5** Meaning of each term and how it is written in the manifest

| No. | Term | Meaning | Written in the manifest as | Section |
| --- | --- | --- | --- | --- |
| 1 | `Tensor[T, s]` | tensor type parameterized by dtype `T` and shape `s` | a tensor's `dtype`, `shape` | [§ 1](#pft) |
| 2 | polymorphic function type | a function type with quantification over its type indices | `forall` | [§ 1](#pft) |
| 3 | type index | a parameter name in a type | `forall`, construction parameters, `let` | [§ 2](#index) |
| 4 | kind | the sort of a type index, which determines its range of values and the operations available on it | `forall: {M: Dim}` | [§ 2](#index) |
| 5 | discriminant | a quantity with finitely many values, used to select a type family branch or to decide whether a tensor exists | expressions in `match`, `optional`, `nullable`, `mutated` | [§ 3](#family) |
| 6 | type family | a function that gives a shape from the value of a discriminant | `signature.types` | [§ 3](#family) |
| 7 | ADT, constructor | a tagged sum type in which each constructor has its own fields | `adts` in `spec/types.yaml` | [§ 4](#adt) |
| 8 | `let` | a named quantity computed from indices | `let: {H_out: "..."}` | [§ 5](#let) |
| 9 | refinement | a predicate that constrains the values of indices | each entry in `shape_rules` | [§ 6](#refinement) |
| 10 | unification | matching the declared shape against the actual shape axis by axis to solve unknown indices | generated from the signature | [§ 7](#unification) |
| 11 | type inference | the process of solving every index at call time | generated from the signature | [§ 7](#unification) |
| 12 | effect | what a call writes to its arguments, and aliasing | `mutated`, `write_only`, `buffer`, `alias` | [§ 8](#effect) |
| 13 | `Maybe[X]`, `present` | an `X` that may be absent; `present(x)` says whether it is given | optional inputs, nullable outputs, `int \| None` parameters | [Extensions § 1](extensions.md#presence) |
| 14 | primitive | a built-in function callable in expressions, such as `broadcast` and `pool.out` | function calls in expressions | [Extensions § 4](extensions.md#let) |
| 15 | generator | a function that generates the values of a metadata tensor at instantiation, such as `prefix_sum(q_lens)` | a tensor's `values` | [Extensions § 7](extensions.md#generators) |
