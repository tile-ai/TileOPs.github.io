# MoE Operators

Every op on this page is used the same way: construct it once, then call it. The
constructor takes what the kernel is compiled with; the call takes the tensors.
Both are documented under each op — `__init__` and `forward`, where `forward` is
what runs when you call `op(...)`.

A routed mixture-of-experts layer is available two ways here. `FusedMoeFwdOp` runs
the whole FFN. The rest are its stages, callable on their own: the op that picks
each token's experts, the ops that move tokens into an expert-contiguous layout and
back, and the expert GEMMs that run on it. `MoeGroupedGemmFwdOp` is one grouped
GEMM; `MoeExpertMLPFwdOp` is the pair of them with the gated activation fused into
the first; `FusedMoEExpertsFwdOp` is that MLP with the permutes around it, on the
tight (no-pad) layout. The routing has to produce the layout the GEMM expects.

## Fused forward

::: tileops.moe.FusedMoeFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Routing and layout

::: tileops.moe.FusedTopKOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.moe.MoePrePermuteFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.moe.MoePermuteAlignFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.moe.MoePostPermuteFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Expert GEMMs

::: tileops.moe.MoeGroupedGemmFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.moe.MoeExpertMLPFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.moe.FusedMoEExpertsFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
