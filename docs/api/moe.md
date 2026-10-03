# MoE Operators

Every op on this page is used the same way: construct it once, then call it. The
constructor takes what the kernel is compiled with; the call takes the tensors.
Both are documented under each op — `__init__` and `forward`, where `forward` is
what runs when you call `op(...)`.

A routed mixture-of-experts layer is available two ways here. `FusedMoEFwdOp` runs
the whole FFN, and `FusedMoESharedExpertFwdOp` adds a shared expert beside the routed ones. The rest are its stages, callable on their own: the op that picks
each token's experts, the ops that move tokens into an expert-contiguous layout and
back, and the expert GEMMs that run on it. `MoEGroupedGemmFwdOp` is one grouped
GEMM; `MoEExpertMLPFwdOp` is the pair of them with the gated activation fused into
the first; `FusedMoEExpertsFwdOp` is that MLP with the permutes around it, on the
tight (no-pad) layout, and `IndexedExpertMLPFwdOp` is the backend it picks instead when
the routes are few enough to read the weights once per route rather than once per expert. `SharedExpertMLPFwdOp` is the dense gated MLP of the shared expert. The routing has to produce the layout the GEMM expects.

## Fused forward

::: tileops.moe.FusedMoEFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.moe.FusedMoESharedExpertFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Routing and layout

::: tileops.moe.FusedTopKFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.moe.MoEPrePermuteFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.moe.MoEPermuteAlignFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.moe.MoEPostPermuteFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Expert GEMMs

::: tileops.moe.MoEGroupedGemmFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.moe.MoEExpertMLPFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.moe.FusedMoEExpertsFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.moe.IndexedExpertMLPFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.moe.SharedExpertMLPFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
