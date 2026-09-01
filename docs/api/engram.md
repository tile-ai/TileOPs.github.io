# Engram Operators

Every op on this page is used the same way: construct it once, then call it. The
constructor takes what the kernel is compiled with; the call takes the tensors.
Both are documented under each op — `__init__` and `forward`, where `forward` is
what runs when you call `op(...)`.

## GateConv

::: tileops.sequence_modeling.EngramGateConvFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.sequence_modeling.EngramGateConvBwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Decode

::: tileops.sequence_modeling.EngramDecodeFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
