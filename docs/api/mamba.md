# Mamba Operators

Every op on this page is used the same way: construct it once, then call it. The
constructor takes what the kernel is compiled with; the call takes the tensors.
Both are documented under each op — `__init__` and `forward`, where `forward` is
what runs when you call `op(...)`.

## Full forward

::: tileops.ops.mamba.mamba2_fwd.Mamba2FwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## SSD stages

::: tileops.ops.mamba.da_cumsum.DaCumsumFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.mamba.cb_producer.CBProducerFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.mamba.ssd_chunk_state.SSDChunkStateFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.mamba.ssd_state_passing.SSDStatePassingFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.mamba.ssd_chunk_scan.SSDChunkScanFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Decode

::: tileops.ops.mamba.ssd_decode.SSDDecodeFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
