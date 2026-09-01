# MoE Operators

Every op on this page is used the same way: construct it once, then call it. The
constructor takes what the kernel is compiled with; the call takes the tensors.
Both are documented under each op — `__init__` and `forward`, where `forward` is
what runs when you call `op(...)`.

A routed mixture-of-experts layer is available two ways here. `FusedMoeFwdOp` runs
the whole FFN. The rest are its stages, callable on their own: the ops that move
tokens into an expert-contiguous layout and back, and the expert GEMMs that run on
it. The GEMMs come in a padded form and a tight one, and the routing has to produce
the layout the GEMM expects.

## Fused forward

::: tileops.moe.FusedMoeFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Routing and layout

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

::: tileops.moe.MoeGroupedGemmNopadFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.moe.MoeGateUpFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.moe.MoeExpertMLPFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.moe.FusedMoEExpertsNopadPersistent3WGFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
