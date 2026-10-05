# Linear Attention Operators

Every op on this page is used the same way: construct it once, then call it. The
constructor takes what the kernel is compiled with; the call takes the tensors.
Both are documented under each op — `__init__` and `forward`, where `forward` is
what runs when you call `op(...)`.

## DeltaNet

::: tileops.linear_attention.DeltaNetChunkFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.linear_attention.DeltaNetChunkBwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.linear_attention.DeltaNetRecurrentFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.linear_attention.DeltaNetInferenceFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Gated DeltaNet

::: tileops.linear_attention.GDNFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Kimi Delta Attention

::: tileops.linear_attention.KDAFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Gated Linear Attention

::: tileops.linear_attention.GLAChunkFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.linear_attention.GLAChunkBwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.linear_attention.GLARecurrentFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.linear_attention.GLAInferenceFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
