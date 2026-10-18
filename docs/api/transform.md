# Transform Operators

An orthogonal transform along one axis. A transform rotates a tensor before it is
quantized; the quantization that follows one is on the
[Quantization](quantization.md) page.

## Walsh-Hadamard

The rotation step of rotation-based quantization, which spreads outliers across
channels so a lower bit width stays accurate.

::: tileops.transform.HadamardTransformFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
