# RoPE Operators

Every op on this page is used the same way: construct it once, then call it. The
constructor takes what the kernel is compiled with; the call takes the tensors.
Both are documented under each op — `__init__` and `forward`, where `forward` is
what runs when you call `op(...)`.

## Base frequencies

One op serves both rotation conventions: pass `is_neox_style=True` to rotate the
two halves of a head against each other, or `is_neox_style=False` to rotate each
adjacent pair.

::: tileops.rope.RoPEFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Scaled frequencies

::: tileops.rope.RoPELlama31FwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.rope.YaRNFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.rope.LongRoPEFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Caller-given cos and sin

The op reads the rotation from `cos` and `sin` rows the caller computes, so any
frequency scheme the caller builds applies.

::: tileops.rope.RoPECosSinFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
