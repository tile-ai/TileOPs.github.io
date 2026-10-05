# Normalization Operators

Every op on this page is used the same way: construct it once, then call it. The
constructor takes what the kernel is compiled with; the call takes the tensors.
Both are documented under each op — `__init__` and `forward`, where `forward` is
what runs when you call `op(...)`.

## LayerNorm

::: tileops.norm.LayerNormFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.norm.FusedAddLayerNormFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## RMSNorm

::: tileops.norm.RMSNormFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.norm.FusedAddRMSNormFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Adaptive LayerNorm

::: tileops.norm.AdaLayerNormFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.norm.AdaLayerNormZeroFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## BatchNorm

::: tileops.norm.BatchNormFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.norm.BatchNormBwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## GroupNorm and InstanceNorm

::: tileops.norm.GroupNormFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.norm.InstanceNormFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
