# Elementwise Operators

Every op on this page is used the same way: construct it once, then call it. The
constructor takes what the kernel is compiled with; the call takes the tensors.
Both are documented under each op — `__init__` and `forward`, where `forward` is
what runs when you call `op(...)`.

## Arithmetic

::: tileops.elementwise.AddFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.SubFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.MulFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.DivFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.FloorDivideFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.RemainderFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.PowFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.MaximumFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.MinimumFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.LerpFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.LerpTensorFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Unary math

::: tileops.elementwise.ExpFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.Expm1FwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.LogFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.Log1pFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.SqrtFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.RsqrtFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.ReciprocalFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.AbsFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.NegFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.SignFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.SinFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.CosFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.ErfFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.FloorFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.CeilFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.RoundFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.TruncFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Activations

::: tileops.elementwise.ReluFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.LeakyReluFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.EluFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.SeluFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.GeluFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.SiluFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.MishFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.SigmoidFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.TanhFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.HardsigmoidFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.HardswishFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.HardtanhFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.SoftplusFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.PreluFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.DropoutFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Gated activations

::: tileops.elementwise.SiluAndMulFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.GeluAndMulFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.GeluTanhAndMulFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Comparison

::: tileops.elementwise.EqFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.NeFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.GtFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.GeFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.LtFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.LeFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.IsnanFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.IsinfFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.IsfiniteFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Logical and bitwise

::: tileops.elementwise.LogicalAndFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.LogicalOrFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.LogicalNotFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.BitwiseAndFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.BitwiseOrFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.BitwiseXorFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.BitwiseNotFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Selection and clamping

::: tileops.elementwise.WhereFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.ClampFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.ClampScalarFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.MaskedFillFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.MaskedFillScalarFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.NanToNumFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Positional encodings

::: tileops.elementwise.AlibiFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.elementwise.SinusoidalFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
