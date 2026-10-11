# Composite ops

A composite op computes through sub-ops and builds none, or only part, of its own kernels.
This page describes:

- how the `Op` base class declares and holds sub-ops;
- how a sub-op's calls are filed under the parent call;
- when sub-ops are built.

## 1. Declaring and holding {#declare}

`MoEExpertMLPFwdOp` is made of two grouped GEMMs:

```python
class MoEExpertMLPFwdOp(Op):
    delegate_types: ClassVar[Mapping[str, type[Op]]] = {
        "gate_up": MoEGroupedGemmFwdOp,
        "down": MoEGroupedGemmFwdOp,
    }

    def __init__(self, layout, activation="silu_and_mul", *, target=None):
        """Configure two grouped GEMMs on ``layout``, the first fusing the gated activation."""
        self.layout = layout
        self.activation = activation
        super().__init__(target=target)
        self.gate_up = self.delegate_for("gate_up", None, layout=layout, activation=activation)
        self.down = self.delegate_for("down", None, layout=layout)
```

The example leaves out the class docstring and `forward`; the full code is in
[`src/tileops/ops/moe/staged.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/ops/moe/staged.py).
Sub-ops follow the same rule as kernels: declared on the class first, then obtained through a
method of the base class.

- **Declaring.** `delegate_types` maps each **stage** name to the sub-op class the composite op
  may hold there; the validator checks it against the manifest entry's composition.
    - The mapping's order is the stage order, the same as the manifest's composition.
    - The composition is therefore a fact about the class, checkable before any call.
- **Holding.** `delegate_for(stage, identity, given=None, /, **params)` is the only way to
  hold a sub-op.
    - One `(stage, identity)` is built once.
    - `identity` covers every argument that changes what the sub-op is built as, the same
      meaning as the identity a kernel's `entry_for` returns.
    - When the construction arguments are settled when the composite op is constructed,
      `identity` is `None`.
- **Inheriting the execution policy.**
    - A sub-op the base class builds inherits the composite op's `target`.
    - A sub-op the caller injects is passed as `given` and held as passed, keeping its own
      `target`.
    - When the composite op is in tuned mode, the base class calls `request_tune()` on each
      newly held sub-op, built or injected.
- **One instance is registered once.**
    - A repeated call with the same `(stage, identity)` returns the instance already held.
    - Registering an instance already held under another `(stage, identity)` makes
      `delegate_for` raise.
- **Derived enumeration.** `held_delegates()`, `iter_kernels()` and `request_tune()` are
  derived from the sub-ops `delegate_for` holds; a composite op does not override them.

## 2. Sub-op calls are filed under stages {#stages}

When a sub-op's call completes:

1. the sub-op reports its record to the parent call on the call stack, step 7 of
   [Construction and calls § The seven steps of a call](lifecycle.md#serve);
1. the parent call files it under the stage `delegate_for` registered, in its own record's
   `stages`.

When a sub-op that `delegate_for` does not hold completes a call, the base class raises and
the parent call fails. Therefore:

- `stages` is either complete, or the call fails;
- a roofline that depends on `stages` is never silently wrong.

## 3. When sub-ops are built {#when}

**Table 1** When sub-ops are built

| No. | The sub-op's construction arguments | Where `delegate_for` is called | Compilation |
| --- | --- | --- | --- |
| 1 | settled at construction | in the constructor, after `super().__init__` | the sub-op is held before tracing |
| 2 | known only at call time | in `forward` | a cold `fullgraph=True` compile is not promised |

Why the second row:

- Building a sub-op runs the sub-op's `Op.__init__`, which dynamo cannot trace.
- Such a composite op can be traced only once every sub-op the call needs is held.
- When traced, the composite op itself is not a node of the graph. When there is no target
  builder, and `forward` calls sub-ops that have compile boundaries, each sub-op's custom op
  is a node of the graph.

`FusedMoEExpertsFwdOp` is on the second row: in `forward` it holds a `pre_permute` sub-op per
expert count of the call, so one instance serves several expert counts.

## 4. Targets and composite ops {#target}

- **When the selected target registered no builder for the composite op:**
    - a composite op that declares no kernel of its own runs its own composition;
    - a composite op that also declares kernels of its own raises `OpNotAvailableError` on the
      call.
- **The sub-ops' targets:** a sub-op the base class builds inherits the composite op's
  `target`; a sub-op injected through `given` keeps its own `target`. Each sub-op selects the
  target that serves it on its own.
- **Undo:** when a call undoes the composite op's target selection, the base class undoes the
  held sub-ops recursively; see [Construction and calls § Failure and undo](lifecycle.md#failure).
