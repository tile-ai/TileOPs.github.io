# Construction and calls

This page describes the steps an op instance goes through inside the `Op` base class, from
construction to each call:

- what `Op.__init__` does;
- through which entry point a call reaches the base class;
- the seven steps of a call;
- how the base class selects a target, installs the implementation table and caches kernels;
- what a failed call undoes.

## 1. Construction: `Op.__init__` {#construct}

A subclass's constructor is written one of two ways, by its direct parent class:

- **Inheriting `Op` directly:** assign the manifest parameters to attributes of the same name,
  then call `super().__init__(target=target)`.
- **Through a family base class:** pass arguments by the family base's signature, for example
  `input_layout` and `base` for a RoPE leaf class, `dim` and `keepdim` for a reduction. The
  family base assigns them and ends by calling `Op.__init__`.

Either way, the manifest parameters' values are settled before `Op.__init__` runs.

`Op.__init__` stores `target`, then takes the five steps below in order. Each step has its
reason to be at construction; none of them selects or builds a kernel.

**Table 1** The steps of `Op.__init__`

| No. | Step | Why at construction |
| --- | --- | --- |
| 1 | Load the backend registry | • It must finish before any traced code<br>• An instance's first call may already be inside `torch.compile`, and dynamo cannot trace module imports and registration<br>• By convention, construction happens outside compilation |
| 2 | Check the parameter values against the manifest, and keep the solved values as `_construction_indices` | • The parameters are settled at construction, so their errors are reported there<br>• Every call uses the solved values, so they are solved once |
| 3 | Install the implementation table and check the contracts, see [§ 4](#target); this also builds the two entry caches `_entries_by_call` and `_built_entries`, both empty | • The result depends on the op class's `kernel_types` and the registry, and does not change for an instance<br>• It resolves classes only and needs no device |
| 4 | Register the instance and get its instance key | • The compile boundary's custom ops look the instance up by this string key<br>• The key must exist before tracing, as a compile-time constant, and is never reused |
| 5 | Build the binding group's `_target_kernels` and the running group's `_delegates`, `_delegate_stages` and `_effect_branches`, all empty; see [Class structure and instance state § Instance state](structure.md#state) | The base class reads and writes these fields without asking whether they exist |

Selecting an implementation and building a kernel wait until the call, when the tensors'
dtypes, shapes and device are known: with in-tree kernels, `kernel_for` selects the
implementation and gets the entry, see [§ 5](#kernel); with a target, the base class calls the
target's builder to build the kernel, see [§ 4](#target).

**Ordering requirements of construction:**

- **The manifest parameters' final values are settled before `super().__init__` is called.**
  `Op.__init__` checks the values the manifest parameters hold when it is called.
    - For example, the `activation` of `FusedMoEFwdOp` and `FusedMoESharedExpertFwdOp` can be
      decided by the injected `experts`.
    - These two ops settle `activation` from the injected object first, then call
      `super().__init__`.
- **Derived attributes that are not manifest parameters may be assigned after
  `super().__init__`.** For example, a pooling op's kernel size expanded per dimension.
- **A composite op builds its sub-ops through `delegate_for` after `super().__init__`.**

**Construction parameters:** construction takes the manifest parameters, the execution policy
`target`, and the extra parameters `injected_parameters` lists. Tuning is not a construction
parameter; it is requested after construction with `request_tune()`. Two kinds of fact the
call's tensors carry are not construction parameters:

1. **The tensors' shapes.** A tensor's dimensions arrive with the call. Taking them again at
   construction would let the instance disagree with the tensors passed.
1. **The input tensors' dtypes.**
    - The base class reads the dtypes from this call's input tensors for the signature check;
      the subclass writes them into the call spec.
    - Which implementation is selected, and whether a built entry is reused, are decided by
      the implementations' selection rules and the identity `entry_for` returns; see
      [§ 5](#kernel).
    - The output dtype is given by the dtype expression in the signature; see
      [Calls and validation § type inference](../manifest/calls.md#inference).

Exceptions and additions:

- **When the manifest declares a dimension or a dtype as a parameter, it is a construction
  parameter.**
    - `AlibiFwdOp`, which has no tensor input, takes `seq_len` and `out_dtype` at
      construction.
    - `ProdFwdOp` takes the accumulation `dtype` at construction, overriding the base class's
      `dtype`, which defaults to `None`.
- **Extra parameters:** the class attribute `injected_parameters` lists the parameter names a
  constructor may take beyond the manifest parameters and the execution policy, such as
  `FusedMoEFwdOp`'s `prepare_finalize` and `experts`. It is only the manifest validator's
  allowlist; the family defines and uses the parameters, and they are not an extension
  mechanism of the base class.

**Construction probes no device:**

- Installing the implementation table resolves classes and reads no device property.
- Call-time tensors arrive after construction, possibly on a device the process has not used
  yet.
- A device that cannot run the op is refused only when a kernel is first selected, built or
  called.
- A construction-time tensor the manifest declares (such as `LongRoPEFwdOp`'s
  `rescale_factors`) is passed at construction, and the construction check reads its shape
  and dtype.

## 2. Call entry points {#entry}

Every eager call that runs an implementation passes through `Op._run_call(inputs, body)`:

- `body` is the subclass's computation `forward`, the one method every op writes;
- how the call reaches `_run_call` depends on whether the subclass has a compile boundary.

**Table 2** Call entry points

| No. | Entry point | Applies to | How it reaches `_run_call` | `body` |
| --- | --- | --- | --- | --- |
| 1 | `op(...)` | ops with a compile boundary | `__call__` calls the generated `_call_boundary`, which reaches `_run_call` through the custom op | `forward` |
| 2 | `op(...)` | ops without a compile boundary | `__call__` binds the arguments by `forward`'s signature, then calls `_run_call` | `forward`, arguments passed by name |
| 3 | an eager `op(...)` on meta tensors | ops with a compile boundary | • Does not reach it<br>• The generated fake runs the signature check, keeps the call record, and builds the result from the signature | — |
| 4 | a traced `op(...)` | ops without a compile boundary | • Does not reach it, and runs no signature check<br>• Calls the target's kernel when a target already serves the instance, and `forward` otherwise<br>• A composite's sub-ops' custom ops become nodes of the graph: tracing runs their fakes, and `_run_call` is reached when the graph's nodes run | — |

Notes on Table 2:

- **Which ops have a compile boundary:** at class definition, the base class generates a
  compile boundary for an op class that meets all three conditions below; `_CompileBoundary`
  registers the custom ops and their names go into the class attribute `compile_op_names`:
    1. it has a manifest entry;
    1. the entry has a call-time tensor input;
    1. the entry has no composition, that is, it is not a composite op.
- **This is a requirement, not an option:** every implemented op that meets the three
  conditions must support `torch.compile(op, fullgraph=True)`, with no per-op exemption. The
  validator checks every implemented entry: the class's `compile_op_names` is non-empty if
  and only if the entry meets the three conditions. The tests further require every such op
  to register a cold `fullgraph=True` compile test as the evidence that it compiles.
- **Choosing between the first two rows:** `__call__` picks row 1 or row 2 of Table 2 by
  whether `compile_op_names` is empty.
- **Details of the compile boundary** (how the custom ops are generated and find the
  instance; the limits on the traced path: with a compile boundary what is traced is the
  generated `_call_boundary`, and for a composite op without one it is `forward`) are in
  [Bringing an op into torch.compile](../../torch-compile.md).
- **One implementation:** a call that runs an implementation has one entry point, so the
  check, the record and the failure handling exist once.

## 3. The seven steps of a call {#serve}

Where the seven steps sit in time is shown in Figure 1 of
[the index page § The lifecycle of an op](index.md#lifecycle).

**Table 3** The steps of a call

| No. | Step | What happens |
| --- | --- | --- |
| 1 | Begin the call | • Push the current thread's call stack<br>• Collect the calls sub-ops complete during this call |
| 2 | Check the signature | Run the checks `_SignaturePlan` generated, which produce this call's record, a `SignatureCall` |
| 3 | Detect an empty write | When there is at least one write and every write holds no elements, run no implementation and build the result from the signature |
| 4 | Select the target | Select one when the instance has none yet and this call needs to run an implementation; see [§ 4](#target) |
| 5 | Run | Run one of the three: the empty-write result, the target's kernel, or `body` |
| 6 | Check the result | Check the return value against the signature |
| 7 | Record | 1. Write the sub-ops' calls into the record's `stages`, by stage<br>2. Pop the call stack<br>3. Keep it as `last_call`<br>4. Report it to the enclosing call |

The signature check and dtypes:

- The signature check decides whether the call satisfies the manifest, dtypes included.
- The dtypes of call-time input tensors are decided by the call, so construction has nothing
  to compare them with; the signature check also uses the dtype indices solved at
  construction, such as a construction-time tensor's dtype.
- A call that passes the check may still be refused when its kernel is selected, for example
  when no implementation supports the dtype; see
  [How an op selects a kernel § How a call finds its kernel](../dispatch/index.md#call-path).

## 4. Targets and the implementation table {#target}

A backend joins TileOPs in one of two ways; how to use them is in
[How a backend joins TileOPs](../dispatch/backends.md#choose). This section describes how the
base class handles them.

**Table 4** The two ways a backend joins

| No. | Way | What it takes over | When the base class handles it |
| --- | --- | --- | --- |
| 1 | `target` | the whole op | selected in step 4 of a call, and unchanged after that |
| 2 | `register_kernel_type` | adds one implementation to a kernel interface, which takes part in selection by its own declared applicability and precedence | merged into the implementation table by `Op.__init__` at construction |

### Selecting the target {#select-target}

The first call that needs to run an implementation selects the target in this order:

1. the `target` passed at construction;
1. when that is `None`, the process default target (set by `set_default_target`);
1. when that is still `None`, detection from the call's device.

Once selected:

- **No backend claims the device:** the in-tree kernels run, and `serving_target` is
  `BUILTIN`.
- **A backend claims it:** the kernel built by the builder this target registered for the op
  runs, and the subclass's `body` does not.
    - The base class calls the builder with arguments in `signature.inputs` order: a
      `TensorSpec` for an input that is present, `None` for an optional input that is not.
    - The manifest parameters on the instance are passed by keyword; see
      [How a backend joins TileOPs § Replace a whole op](../dispatch/backends.md#target).
- **The selected target registered no builder for the op:**
    - an op that declares kernels of its own raises `OpNotAvailableError` and does not fall
      back to the in-tree implementation;
    - a composite op that declares no kernel runs its own composition.
- `serving_target` returns the selection, and `None` before one is made.

The call's device is settled by the signature check, taken from the first of:

1. the device of the call-time tensors;
1. the declared `device` parameter;
1. the device of the construction-time tensors;
1. when there is none of the three, the current CUDA device.

When CUDA is unavailable too, and there is no constructor argument or process default target,
the base class has nothing to select from:

- this call runs the in-tree body;
- the instance still has no target, and the next call selects again.

The cache of kernels a target builds:

- the key is the call's device plus each input's dtype and shape;
- an optional input that is not passed holds its own slot;
- output buffers are not part of the key.

Before a target is called, the generated checks have run, and the base class guarantees three
things:

1. every tensor is on the call's device, except tensors declaring `device: cpu`;
1. every input this call does not write is contiguous, and a written input is contiguous when
   it declares `contiguous: true`;
1. the call, including the output buffers the caller provides, satisfies the signature.

For an op with no call-time tensor input, the device follows the rules in
[Calls and validation § Call device](../manifest/calls.md#device).

### Installing the implementation table {#install}

The implementation table `_installed_kernel_types` records the implementation class of each
key. `Op.__init__`, in order:

1. merges `kernel_types` with the kernel types backends registered for this op; keys must not
   repeat;
1. checks that the implementations follow their kernel interfaces' contracts, raising
   `TypeError` or `ValueError` at construction otherwise:
    - each inherits both `Kernel` and the kernel interface of its key;
    - `entry_for` is a classmethod;
    - `forward` takes the interface's arguments;
    - every key belongs to some interface;
    - each interface has at most one `general` implementation;
    - `preferred_over` names only other implementations of the same interface, and the
      `general` implementation declares none;
    - `preferred_over` forms no cycle.

A composite op that declares no kernel of its own has an empty implementation table; its
sub-ops each install their own.

## 5. Kernel cache {#kernel}

This section distinguishes three words:

- **A kernel type** is a subclass of `Kernel`, the "implementation" of
  [How an op selects a kernel § Terms](../dispatch/index.md#terms). It declares which devices
  it runs on, which calls it applies to, and how it is built.
- **An entry** is the object a kernel type's `entry_for` builds for a call. `kernel_for`
  returns it, and the subclass calls it to compute.
- **A kernel** is a callable object: a `Kernel` instance in an entry, of which an entry holds
  one or more, or the callable a target's builder returns. A `Kernel` instance holding a
  TileLang program may build the program only at its first launch.

One kernel type builds several entries, one per identity. Taking `RMSNormOnChipKernel` as the
example:

**Table 5** Kernel type and entry

| No. | | Kernel type | Entry |
| --- | --- | --- | --- |
| 1 | Example | the class `RMSNormOnChipKernel` | an object built for `n=4096`, fp16 and other arguments |
| 2 | Exists from | module import | built when the first call that needs it arrives |
| 3 | Carries | its availability and applicability declarations (`supported_archs`, `refusal`, `preferred_over`) and `entry_for` | its construction arguments (settled by the identity) and one or more kernels |
| 4 | How many | one per key | under each interface, at most one per `(kernel type, identity)`, for example one for `n=4096`, fp16 and another for `n=8192`, bf16 |
| 5 | Kept in | `kernel_types`, `_installed_kernel_types` | `_built_entries` |
| 6 | Used for | the base class selects this call's implementation from it | the subclass calls it: `entry(x, weight)` |

`register_kernel_type` adds a kernel type to the candidates and builds no entry;
`register_kernel_builder` registers a builder that builds the target's kernel directly,
without selecting among kernel types; see [§ 4](#target).

A subclass gets a kernel through `self.kernel_for(interface, call)`:

- how to write the arguments is in
  [Adding a new op § kernel_for](../../new-op.md#kernel-selection);
- kernel interfaces, implementations, call specs and entries are defined in
  [How an op selects a kernel § Terms](../dispatch/index.md#terms).

`kernel_for` takes these steps in order:

1. **Confirm the interface.**
    - When `interface` is not in `interfaces`, it raises `OpNotAvailableError`.
    - When the call spec gives no device and CUDA is available, the base class fills in the
      current CUDA device, so the cache key names a device.
1. **Look up by call.**
    - `_entries_by_call` is keyed by `(interface, call)`.
    - Equal call specs describe the same call: a hit is one lookup and reads no device
      property.
1. **Check the call spec.** On a miss, it raises `TypeError` unless all of these hold:
    - the call spec is the type the interface's `request` declares;
    - every field can key a cache;
    - it states no device facts of its own.
1. **Look up by build.**
    - Dispatch selects one implementation; see
      [How an op selects a kernel § How a call finds its kernel](../dispatch/index.md#call-path).
      `key_for(interface, call)` returns the key this step selects.
    - The implementation's `entry_for(call)` gives the identity and the builder.
    - `_built_entries` is split by interface and keyed by `(implementation class, identity)`
      within each: calls with the same identity share one entry, and a new identity builds a
      new entry and keeps it.
    - When the call spec's device is a CUDA device, building the entry and handling a tuning
      request both run under `torch.cuda.device(call.device)`.

Further notes:

- **Dtypes and entries:** call specs with different dtypes are dispatched separately; whether
  a built entry is reused is decided by the identity the selected implementation's
  `entry_for` returns.
- **Only the implementation knows the identity:** a subclass defines no identity and keeps no
  cache of its own.
- **Only the base class holds entries:** in its body, a subclass receives the entry
  `kernel_for` returns in a local variable and does not store it on the instance; the next
  call gets it through `kernel_for` again, and a cache hit is one lookup.

## 6. Failure and undo {#failure}

When a call raises an `Exception`:

- the call stack is popped;
- this call does not become `last_call`, and the previous record stays.

If the target was selected in this call's step 4:

- `_reset_binding` resets the binding group to the initial values `Op.__init__` gave it;
- the kernels this call built are dropped with it;
- the same is done to every held sub-op, recursively.

After that, the binding group is back to having no target selected; the records of completed
calls and the held sub-ops are kept. The next call selects the target again.

**Table 6** Instance state after a failure

| No. | Case | Target | Built kernels | `last_call` |
| --- | --- | --- | --- | --- |
| 1 | the target was selected in this call | undone | dropped | unchanged |
| 2 | the target was selected in an earlier call | kept | kept | unchanged |

In addition:

- what has already been written into tensors is not undone;
- exceptions that do not inherit `Exception`, such as `KeyboardInterrupt`, skip this handling.
