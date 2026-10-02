# Quantization and Dequantization Operators

Every op on this page is used the same way: construct it once, then call it. The
constructor takes what the kernel is compiled with; the call takes the tensors.
Both are documented under each op — `__init__` and `forward`, where `forward` is
what runs when you call `op(...)`.

## INT8 quantization

::: tileops.quantization.INT8QuantPerTensorFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.quantization.INT8QuantPerChannelFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.quantization.INT8QuantPerBlockFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.quantization.SmoothQuantFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## FP8 quantization

::: tileops.quantization.FP8QuantFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.quantization.FP8QuantPerBlockFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## INT4 quantization

::: tileops.quantization.INT4QuantPerGroupFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## INT8 dequantization

::: tileops.quantization.INT8DequantPerTensorFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.quantization.INT8DequantPerChannelFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.quantization.INT8DequantPerBlockFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
