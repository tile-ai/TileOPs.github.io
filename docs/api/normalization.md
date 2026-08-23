# Normalization Operators

Every op on this page is used the same way: construct it once, then call it. The
constructor takes what the kernel is compiled with; the call takes the tensors.
Both are documented under each op — `__init__` and `forward`, where `forward` is
what runs when you call `op(...)`.

## Layer norm

::: tileops.ops.norm.layer_norm.LayerNormFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.norm.fused_add_layer_norm.FusedAddLayerNormFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## RMS norm

::: tileops.ops.norm.rms_norm.RMSNormFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.norm.fused_add_rms_norm.FusedAddRMSNormFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Adaptive layer norm

::: tileops.ops.norm.ada_layer_norm.AdaLayerNormFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.norm.ada_layer_norm_zero.AdaLayerNormZeroFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Batch norm

::: tileops.ops.norm.batch_norm.BatchNormFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.norm.batch_norm.BatchNormBwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Group and instance norm

::: tileops.ops.norm.group_norm.GroupNormFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.norm.instance_norm.InstanceNormFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
