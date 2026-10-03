# RoPE Operators

Every op on this page is used the same way: construct it once, then call it. The
constructor takes what the kernel is compiled with; the call takes the tensors.
Both are documented under each op — `__init__` and `forward`, where `forward` is
what runs when you call `op(...)`.

## Base frequencies

One op serves both rotation conventions: pass `rope_layout` as `"neox"` to
rotate the two halves of a head against each other, or `"interleaved"` to rotate
each adjacent pair.

::: tileops.rope.RopeFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.rope.RopeNeoxPositionIdsFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Scaled frequencies

::: tileops.rope.RopeLlama31FwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.rope.RopeYarnFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.rope.RopeLongRopeFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
