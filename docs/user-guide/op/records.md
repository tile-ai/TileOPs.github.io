# Call records and tuning

This page describes what the `Op` base class offers after a call:

- the record of the last call;
- the roofline evaluated from the record;
- the enumeration of built kernels;
- tuning.

## 1. last_call {#last-call}

`op.last_call` returns the record of the last completed call, a `SignatureCall`:

- before any call has completed, it raises `RuntimeError`;
- a failed call does not replace the record; see
  [Construction and calls § Failure and undo](lifecycle.md#failure).

**Table 1** Fields of `SignatureCall`

| No. | Field | Meaning |
| --- | --- | --- |
| 1 | `indices` | The values solved for the signature's indices: `forall`'s scalar and dtype indices, construction parameters in index positions, and `let`; a `forall` index of type `Seq[Int]` is not among them |
| 2 | `branch_key` | The discriminant branch the signature check took |
| 3 | `device` | The call's device, settled by the signature check; the base class selects the target from it |
| 4 | `tensors` | • The shape and dtype of each input, construction-time tensor and output present in this call, by its name in the signature<br>• An `out` the caller passes is not listed separately |
| 5 | `traffic` | How many times each tensor is read and written, given by the manifest's effect rules |
| 6 | `written`, `has_out` | The inputs this call writes, and whether the caller passed `out` |
| 7 | `metadata` | The metadata input tensors that declare `values` |
| 8 | `stages` | The calls the composite op's sub-ops completed in each stage, in completion order; see [Composite ops](composite.md#stages) |

When using a record:

- **Treat the record as immutable.** `stages` is written once, before the record becomes
  `last_call`.
- **`metadata` holds references to the tensors, not copies.**
    - `metadata_values(name)` reads the tensor's contents only when it is called, so the call
      needs no synchronization to the host.
    - If the caller later changes those tensors, what an earlier record reads changes with
      them.

## 2. Roofline {#roofline}

- **`eval_roofline()`**
    - Generated from the manifest entry's `roofline` field, it evaluates `last_call` and
      returns `(flops, bytes)`.
    - The formula syntax and what is counted are in the design document
      [Roofline](../../design/roofline.md).
- **`roof_key()`**
    - Names the compute unit that prices those FLOPs; the base default is `"cuda_core.fp32"`.
    - A subclass whose FLOPs are matrix multiplications overrides it, usually returning the
      `tensor_core_roof` of the input dtype read from `self.last_call`; see the design
      document [Roofline § 1.4](../../design/roofline.md#14-compute-roof).

A subclass may also override two methods:

1. **`eval_roofline_read_bytes()`** returns the read part of `bytes`, for NCU's bytes audit.
    - The base class subtracts the writes the signature settles from `bytes`.
    - When the signature cannot settle the read part, a subclass overrides it and returns
      `None`.
1. **`roofline_data_terms()`** returns the quantities through which the inputs' values decide
   the FLOPs or bytes.
    - For example, the keys a DSA call selected and the distinct `kv` rows it reached.
    - The benchmark records them beside the reading; they are not part of `(flops, bytes)`.
    - The base class returns an empty dict.

## 3. Enumeration {#enumerate}

**Table 2** Enumeration methods

| No. | Method | Returns |
| --- | --- | --- |
| 1 | `iter_kernels()` | • Every `Kernel` in each interface's entries and in the held sub-ops, each once<br>• Kernels a target built are not walked; `built_entries(interface)` shows them |
| 2 | `built_entries(interface)` | • In-tree path: the entries built for this kernel interface, told apart by `(implementation class, identity)`<br>• When a target serves the op: every target kernel of the op, told apart by device and each input's dtype and shape; `interface` does not filter |
| 3 | `held_delegates()` | The sub-ops the composite op holds, in stage order |

What the three methods are for:

- `built_entries()` is for tests and benchmark reports only.
- `iter_kernels()` also serves `request_tune()` and `kernel_config()`.
- `held_delegates()` also serves the recursive undo of sub-ops after a failure.

The call path does not get kernels through these three methods:

- the in-tree body gets them through `kernel_for`; on the target path the base class looks
  them up by the cache key;
- both select and build a kernel on a cache miss, and do not raise because of the miss;
- they still raise when the call or the implementation breaks a requirement; see
  [Construction and calls § Kernel cache](lifecycle.md#kernel).

`iter_kernels()` enumerates only the declared places and does not scan attributes by
reflection:

- the places it enumerates: each interface's entries and the held sub-ops;
- a reflective scan would miss, without an error, kernels nested deeper or kept in an
  attribute of a type it does not recognize;
- enumerating only declared places makes an omission show up as a missing declaration.

## 4. Tuning {#tune}

- **`request_tune()`:** the only way into tuned mode; neither ops nor kernels take a `tune`
  construction parameter. It puts the op and the sub-ops it holds in tuned mode.
    - It calls `Kernel.request_tune()` on every built kernel: a kernel whose program is built
      tunes now, and one whose program is not built yet tunes at the launch that builds it; a
      kernel without `autotune_configs` is not tuned.
    - For an entry built afterwards, the base class calls `request_tune()` on each `Kernel` in
      it right after building it.
- **Targets:** a tuning request cannot reach the kernels a target builds, so the base class
  warns, once per instance.
- **Subclass overrides:** the above is the base class's behavior, and a subclass may override
  it. `GemmW4A16FwdOp` does not support generic tuning: it overrides `request_tune()` to warn
  and use its own calibrated selection.
- **`kernel_config()`:** returns the op's own configuration; without one, the configuration of
  the first configured kernel `iter_kernels()` yields.
