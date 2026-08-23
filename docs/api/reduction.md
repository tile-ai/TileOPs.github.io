# Reduction Operators

Every op on this page is used the same way: construct it once, then call it. The
constructor takes what the kernel is compiled with; the call takes the tensors.
Both are documented under each op — `__init__` and `forward`, where `forward` is
what runs when you call `op(...)`.

## Sums, means and extrema

::: tileops.ops.reduction.reduce.SumFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.reduction.reduce.MeanFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.reduction.reduce.ProdFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.reduction.reduce.AmaxFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.reduction.reduce.AminFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Variance and deviation

::: tileops.ops.reduction.reduce.VarFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.reduction.reduce.VarMeanFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.reduction.reduce.StdFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Arg reductions

::: tileops.ops.reduction.argreduce.ArgmaxFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.reduction.argreduce.ArgminFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Softmax

::: tileops.ops.reduction.softmax.SoftmaxFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.reduction.softmax.LogSoftmaxFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.reduction.softmax.LogSumExpFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Vector norms

::: tileops.ops.reduction.vector_norm.L1NormFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.reduction.vector_norm.L2NormFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.reduction.vector_norm.InfNormFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Cumulative

::: tileops.ops.reduction.cumulative.CumulativeOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.reduction.cumulative.CumsumFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.reduction.cumulative.CumprodFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Logical reductions

::: tileops.ops.reduction.logical_reduce.AllFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.reduction.logical_reduce.AnyFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.reduction.logical_reduce.CountNonzeroFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
