# Linear Attention Operators

Every op on this page is used the same way: construct it once, then call it. The
constructor takes what the kernel is compiled with; the call takes the tensors.
Both are documented under each op — `__init__` and `forward`, where `forward` is
what runs when you call `op(...)`.

## DeltaNet

::: tileops.ops.linear_attention.deltanet.DeltaNetOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.linear_attention.deltanet.DeltaNetFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.linear_attention.deltanet.DeltaNetBwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.linear_attention.deltanet_recurrence.DeltaNetDecodeFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Gated DeltaNet

::: tileops.ops.linear_attention.gated_deltanet.GatedDeltaNetOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.linear_attention.gated_deltanet.GatedDeltaNetBTHDFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.linear_attention.gated_deltanet.GatedDeltaNetBHTDFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.linear_attention.gated_deltanet.GatedDeltaNetPrefillBTHDFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.linear_attention.gated_deltanet.GatedDeltaNetPrefillBHTDFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.linear_attention.gated_deltanet.GatedDeltaNetDecodeFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.linear_attention.gated_deltanet.GatedDeltaNetBwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Gated linear attention

::: tileops.ops.linear_attention.gla.GLAFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.linear_attention.gla.GLABwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.linear_attention.gla_recurrence.GLADecodeFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
