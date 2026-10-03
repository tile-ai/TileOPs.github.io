# Transform Operators

An orthogonal transform along one axis. A transform rotates a tensor before it is
quantized; the quantization that follows one is on the
[Quantization](quantization.md) page.

## Walsh-Hadamard

`HadamardTransformFwdOp` is specified and not yet implemented, so it has no
constructor to document here. The signature it will serve is in
`src/tileops/manifest/spec/transform.yaml`: a fast Walsh-Hadamard transform along
the last axis scaled by $1/\sqrt{n}$, where $n$ is `base_order` times a power of
two, in float16 or bfloat16 with the butterfly accumulated in float32. It is the
rotation step of rotation-based quantization, which spreads outliers across
channels so a lower bit width stays accurate.

This page gains the op's `__init__` and `forward` when a kernel serves it.
