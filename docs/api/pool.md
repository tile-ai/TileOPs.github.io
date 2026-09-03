# Pooling Operators

Every op on this page is used the same way: construct it once, then call it. The
constructor takes what the kernel is compiled with; the call takes the tensors.
Both are documented under each op — `__init__` and `forward`, where `forward` is
what runs when you call `op(...)`.

## Average pooling

::: tileops.pool.AvgPool1dFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.pool.AvgPool2dFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.pool.AvgPool3dFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Max pooling

::: tileops.pool.MaxPool1dFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.pool.MaxPool1dIndicesFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.pool.MaxPool2dFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.pool.MaxPool2dIndicesFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.pool.MaxPool3dFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.pool.MaxPool3dIndicesFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Adaptive pooling

::: tileops.pool.AdaptiveAvgPool2dFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.pool.AdaptiveMaxPool2dFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.pool.AdaptiveMaxPool2dIndicesFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Chunked sequence mean

The one op on this page with no PyTorch counterpart. It averages the sequence axis of
a `[batch, seq, heads, dim]` tensor in fixed-size chunks, and can follow ragged
sequence boundaries rather than a uniform split.

::: tileops.pool.MeanPoolingFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
