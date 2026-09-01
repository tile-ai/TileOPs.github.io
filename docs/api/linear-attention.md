# Linear Attention Operators

Every op on this page is used the same way: construct it once, then call it. The
constructor takes what the kernel is compiled with; the call takes the tensors.
Both are documented under each op — `__init__` and `forward`, where `forward` is
what runs when you call `op(...)`.

## DeltaNet

::: tileops.linear_attention.DeltaNetOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.linear_attention.DeltaNetFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.linear_attention.DeltaNetBwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.linear_attention.DeltaNetDecodeFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Gated DeltaNet

::: tileops.linear_attention.GatedDeltaNetOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.linear_attention.GatedDeltaNetBTHDFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.linear_attention.GatedDeltaNetBHTDFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.linear_attention.GatedDeltaNetPrefillBTHDFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.linear_attention.GatedDeltaNetPrefillBHTDFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.linear_attention.GatedDeltaNetDecodeFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.linear_attention.GatedDeltaNetBwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Gated linear attention

::: tileops.linear_attention.GLAFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.linear_attention.GLABwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.linear_attention.GLADecodeFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
