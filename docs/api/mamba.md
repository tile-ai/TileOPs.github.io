# Mamba Operators

Every op on this page is used the same way: construct it once, then call it. The
constructor takes what the kernel is compiled with; the call takes the tensors.
Both are documented under each op — `__init__` and `forward`, where `forward` is
what runs when you call `op(...)`.

## Full forward

::: tileops.mamba.Mamba2FwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## SSD stages

::: tileops.mamba.DaCumsumFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.mamba.CBProducerFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.mamba.SSDChunkStateFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.mamba.SSDStatePassingFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.mamba.SSDChunkScanFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Decode

::: tileops.mamba.SSDDecodeFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
