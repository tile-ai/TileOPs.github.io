# Op base class developer guide

This guide describes TileOPs' `Op` base class:

- what it does for every op;
- what an op subclass declares to it;
- the steps construction and a call go through inside the base class.

It is written for developers who change the `Op` base class and the code generated for it,
and for developers who need to know how the op layer works inside.

## 1. Scope of this guide {#scope}

This guide covers the `Op` base class only. The topics below each have a guide of their own;
where this guide uses one of their conclusions it links to it rather than repeating it.

**Table 1** Related guides

| No. | Topic | Guide |
| --- | --- | --- |
| 1 | • The steps to add an op: the manifest entry, the op class's members, the arguments of `kernel_for`<br>• Registration, tests and benchmarks | [Adding a new op](../../new-op.md) |
| 2 | • Calling an op inside `torch.compile`<br>• When a compile boundary is generated, and the cold-compile requirement<br>• How the custom op and its fake are generated | [Bringing an op into torch.compile](../../torch-compile.md) |
| 3 | • Selecting among kernel interfaces and implementations<br>• Adding a kernel<br>• How a backend joins | [How an op selects a kernel](../dispatch/index.md) |
| 4 | • The format of a manifest entry<br>• The checks generated from a signature | [Reading and writing the manifest](../manifest/index.md) |

## 2. Op and Kernel {#op-kernel}

An operator that runs kernels of its own is split into two classes:

1. **Op** checks the inputs, selects the kernel and assembles the outputs.
    - An op class with a manifest entry has the same name as the entry's key.
    - An intermediate base class that concrete ops inherit from may have no entry.
1. **Kernel** is the callable implementation object. It holds the TileLang program that runs
   on the device and its tile configuration; the program may be built only at the first
   launch.

A composite op may hold only sub-ops and declare no kernel of its own; see
[Composite ops](composite.md).

An op instance is constructed once and called many times:

- **Passed at construction:**
    - the parameters the manifest declares;
    - the execution policy `target`;
    - for a few ops, extra parameters their family defines, described below.
- **Known only at each call:** the shapes and dtypes of the tensors the call passes. The call
  decides them, so they are not construction parameters.
    - For an op with no tensor input, construction parameters give this information.
    - A tensor the manifest declares among the parameters (a construction-time tensor) is
      passed at construction, such as `LongRoPEFwdOp`'s `rescale_factors`; the construction
      check uses its shape and dtype.
- **Kernels:** built the first time a specialization is needed, then cached, so the same
  specialization is reused afterwards. Whether an in-tree entry is reused is decided by the
  identity the implementation's `entry_for` returns; a kernel a target builds is cached by
  the call's device and each input's dtype and shape. See
  [Construction and calls § Kernel cache](lifecycle.md#kernel).

**Extra construction parameters:** the class attribute `injected_parameters` is the manifest
validator's allowlist.

- Of the constructor parameters the manifest does not declare, only those listed here are
  allowed, besides the execution policy `target`.
- The family defines and uses these parameters; they are not an extension mechanism of the
  base class, which only lets their names through.
- Only MoE's `FusedMoE` uses it today, listing `prepare_finalize` and `experts`.

## 3. The lifecycle of an op {#lifecycle}

This guide distinguishes three subjects:

1. **op class**: a subclass of `Op`, such as `RMSNormFwdOp`.
1. **op instance**: an object an op class constructs.
1. **a call**: one call of an op instance.
    - An op instance is callable: `Op` defines `__call__`.
    - `op(x, weight)` calls `Op.__call__`, which enters the subclass's `forward`.
    - `Op` does not inherit `torch.nn.Module`.

An op passes through three stages in order:

**Table 2** The three stages of an op

| No. | Stage | Subject | Trigger | How often |
| --- | --- | --- | --- | --- |
| 1 | Class definition | op class | • Importing the op's module runs the `class` statement<br>• `Op.__init_subclass__` runs | Once per op class |
| 2 | Construction | op instance | • `op = RMSNormFwdOp(normalized_shape=(4096,))`<br>• The subclass's `__init__` runs and calls `Op.__init__` | Once per op instance |
| 3 | Call | a call | • `y = op(x, weight)`<br>• `Op.__call__` runs | Every call |

Figure 1 strings the key steps of the three stages together in time order, with the call
stage split into running and ending.

![The lifecycle of an op](img/lifecycle.svg)

**Figure 1** One op, from class definition to the end of a call.

How to read Figure 1:

- **Step numbers:** "step n" of construction refers to Table 1 of
  [Construction and calls § Construction](lifecycle.md#construct); "step n" of the call and
  end stages refers to Table 3 of
  [Construction and calls § The seven steps of a call](lifecycle.md#serve).
- **An `opt` box** runs only when its condition holds; **an `alt` box** runs one of its
  branches by condition.
- **`kernel_for` is drawn only as far as "build the entry".** How the implementation is
  selected is in
  [How an op selects a kernel § How a call finds its kernel](../dispatch/index.md#call-path).
- **The figure shows only eager calls on ordinary tensors.** An eager call on meta tensors of
  an op with a compile boundary runs the signature check in the generated fake and does not
  enter `_run_call`. When traced, an op without a compile boundary skips the signature check,
  and an op with one runs the check in its fake, again without entering `_run_call`. Both are
  in [Construction and calls § Call entry points](lifecycle.md#entry).

**Table 3** What the base class and the subclass do at each stage

| No. | Stage | The base class does | The subclass provides |
| --- | --- | --- | --- |
| 1 | Class definition | • Generates the signature check and output-shape inference from the manifest entry, and `eval_roofline` when the entry has a `roofline`<br>• Generates custom ops, the compile boundary, when the entry has a call-time tensor input and no composition<br>• Records the names of the manifest parameters, by which a target's builder is called | • An op class with a manifest entry has the entry key's name; an intermediate base class may have no entry<br>• Declares `kernel_types`, `interfaces`, `delegate_types` |
| 2 | Construction | `Op.__init__` stores `target`, then in order:<br>1. loads the backend registry<br>2. checks the parameter values against the manifest<br>3. installs the implementation table and checks the contracts<br>4. registers the instance<br>5. builds the instance state fields | • Inheriting `Op` directly: assigns the manifest parameters to attributes of the same name, then calls `super().__init__(target=target)`<br>• Through a family base class: passes arguments by the family base's signature, which ends by calling `Op.__init__`<br>• Builds derived attributes and sub-ops after that |
| 3 | Call: before the body | 1. Checks the call against the manifest signature.<br>2. If every tensor the call writes holds no elements, runs no kernel and returns the result built from the signature.<br>3. If the instance has no target yet, selects one from the call's device. | — |
| 4 | Call: running | • For an empty write, builds the result from the signature<br>• Otherwise runs the target's kernel when there is a target builder, and the subclass's body when there is not<br>• When the body calls `kernel_for`: looks up the cache, and on a miss selects the implementation and builds the entry<br>• When the body calls `delegate_for`: builds or returns the held sub-op | • `forward`<br>• Calls `kernel_for(interface, call)` and `delegate_for(stage, identity, ...)` in it |
| 5 | Call: after the body returns | • Checks the return value<br>• Files the sub-ops' calls under their stages<br>• Keeps the call record `last_call` | — |
| 6 | Call fails | • Discards this call's record; `last_call` is unchanged<br>• When this call selected the target: undoes the target, drops the built kernels, and undoes the sub-ops' bindings level by level<br>• When an earlier call selected the target: keeps the target, the kernels and the sub-ops | — |

Notes on Table 3:

- **The call rows describe eager calls.** Traced inside `torch.compile`, an op without a
  compile boundary skips the signature check; see
  [Construction and calls § Call entry points](lifecycle.md#entry).
- **Construction does not touch kernels.** It selects and builds no kernel and reads no
  device property.
- **Selecting an implementation and building a kernel happen at call time**, once the
  tensors' dtypes, shapes and device are known:
    - when in-tree kernels run, `kernel_for` selects the implementation and gets the entry;
    - when a target serves the op, the base class calls the target's builder to build the
      kernel and caches it.
- Why each step of `Op.__init__` is at construction is in
  [Construction and calls § Construction](lifecycle.md#construct).

Two more duties belong to no single call:

1. **Roofline.**
    - `eval_roofline()` evaluates the record of the last call.
    - A subclass whose FLOPs do not run on fp32 CUDA cores overrides `roof_key()`.
    - See [Call records and tuning § Roofline](records.md#roofline).
1. **Tuning and enumeration.**
    - `request_tune()` is the only way into tuned mode: it puts the op and its sub-ops in
      tuned mode and sends a tuning request to the built kernels `iter_kernels()` yields;
      in-tree entries built afterwards are tuned too; when the request cannot reach a target,
      it warns.
    - `iter_kernels()` and related methods enumerate the built kernels.
    - See [Call records and tuning](records.md).

Three rules for subclasses follow:

1. **Do not repeat checks.** A subclass does not repeat checks the signature declares, and
   does not check the device type.
    - When the body runs, the signature check has passed.
    - A kernel declares the devices it runs on.
1. **Do not build your own cache.** A subclass keeps no kernel cache and does not decide
   whether to build a kernel by whether some attribute is empty; `kernel_for` owns the cache.
1. **Hold sub-ops only through `delegate_for`.** When a sub-op that is not held completes a
   call, the parent call fails.

## 4. Members of the base class {#members}

The base class's members fall into three layers by who uses them:

1. **External interface:** used by code that calls an op; see Table 4.
1. **Internal interface:** declared or called by op subclasses; see Table 5.
1. **Implementation:** a subclass neither calls nor overrides these; see the end of this
   section.

**Table 4** External interface: used by code that calls an op

| No. | Member | Purpose | Details |
| --- | --- | --- | --- |
| 1 | Construction parameter `target` | The execution policy, which every op takes | [Construction and calls § Targets and the implementation table](lifecycle.md#target) |
| 2 | `__call__` | • Calls the op: the caller writes `op(...)`<br>• The subclass's `forward` decides the parameters; the caller never calls `forward` directly | [Construction and calls § Call entry points](lifecycle.md#entry) |
| 3 | `serving_target` | The target this instance selected | [Construction and calls § Targets and the implementation table](lifecycle.md#target) |
| 4 | `last_call`, `eval_roofline()` | The last completed call, and its `(flops, bytes)` | [Call records and tuning § last_call](records.md#last-call) |
| 5 | `request_tune()`, `kernel_config()` | Tuning, and reading the configuration in use | [Call records and tuning § Tuning](records.md#tune) |
| 6 | `iter_kernels()`, `built_entries(interface)`, `held_delegates()` | Enumerate the built kernels, entries and sub-ops | [Call records and tuning § Enumeration](records.md#enumerate) |

**Table 5** Internal interface: declared or called by op subclasses

| No. | Member | Purpose | Details |
| --- | --- | --- | --- |
| 1 | `Op.__init__(*, target=None)` | • Called as `super().__init__(...)` once the subclass has assigned its manifest parameters<br>• Derived attributes and sub-ops are built after it | [Construction and calls § Construction](lifecycle.md#construct) |
| 2 | `kernel_types`, `interfaces` | The subclass's kernel interfaces and implementations | [Construction and calls § Kernel cache](lifecycle.md#kernel) |
| 3 | `injected_parameters` | • The manifest validator's allowlist: the parameter names a constructor may take beyond the manifest parameters and the execution policy<br>• The family defines and uses the parameters; they are not an extension mechanism of the base class | [Construction and calls § Construction](lifecycle.md#construct) |
| 4 | `dtype` | • Declared on the base class with default `None`<br>• A subclass taking a `dtype` construction parameter overrides it | [Construction and calls § Construction](lifecycle.md#construct) |
| 5 | `forward` | The computation, the one method every op writes; with a compile boundary, the base class runs it inside the custom op through the generated `_call_boundary` | [Construction and calls § Call entry points](lifecycle.md#entry) |
| 6 | `kernel_for(interface, call)`, `key_for(interface, call)` | • `kernel_for` gets the entry that serves this call<br>• `key_for` returns only the selected implementation's key and builds no entry; tests use it to check the selection | [Construction and calls § Kernel cache](lifecycle.md#kernel) |
| 7 | `delegate_types`, `delegate_for(stage, identity, ...)` | Declare and hold sub-ops | [Composite ops](composite.md) |
| 8 | `roof_key()`, `eval_roofline_read_bytes()`, `roofline_data_terms()` | The parts of the roofline a subclass may override | [Call records and tuning § Roofline](records.md#roofline) |

**Implementation:**

- Includes `_run_call`, `_reset_binding`, the call stack, `_SignaturePlan`, which generates
  the signature checks, and `_CompileBoundary`, which generates the compile boundary.
- These members start with an underscore; a subclass neither calls nor overrides them.
- See [Construction and calls](lifecycle.md) and
  [Class structure and instance state](structure.md).

**Members generated from the manifest entry, which a subclass does not write:** the signature
check, `_infer_output_shapes`, `eval_roofline` (when the entry has a `roofline`),
`_call_boundary` (with a compile boundary), and the compile boundary's custom ops.

## 5. Pages of this guide {#pages}

**Table 6** Pages

| No. | Page | Contents |
| --- | --- | --- |
| 1 | [Construction and calls](lifecycle.md) | • `Op.__init__`<br>• Call entry points, the seven steps of a call<br>• Targets and the implementation table<br>• Kernel cache<br>• Failure and undo |
| 2 | [Composite ops](composite.md) | • Declaring and holding sub-ops<br>• How sub-op calls are filed under the parent call |
| 3 | [Call records and tuning](records.md) | • `last_call`<br>• Roofline<br>• Enumeration<br>• Tuning |
| 4 | [Class structure and instance state](structure.md) | • Class diagram<br>• Structure of the generated code<br>• Grouping of instance state<br>• What the base class guarantees when something is added |
