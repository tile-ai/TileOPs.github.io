# Attention Operators

Every op on this page is used the same way: construct it once, then call it. The
constructor takes what the kernel is compiled with; the call takes the tensors.
Both are documented under each op — `__init__` and `forward`, where `forward` is
what runs when you call `op(...)`.

## Multi-Head Attention

::: tileops.attention.MHADecodePagedWithKVCacheFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Grouped-Query Attention

::: tileops.attention.GQABwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.attention.GQADenseFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.attention.GQAPagedFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.attention.GQAPrefillPagedWithKVCacheFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.attention.GQAVarlenFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Multi-Head Latent Attention

::: tileops.attention.MLADecodeWithKVCacheFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Native Sparse Attention

::: tileops.attention.NSACompressedVarlenFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.attention.NSATopKVarlenFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.attention.NSAVarlenFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## DeepSeek Sparse Attention

::: tileops.attention.DSADecodeWithKVCacheFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Attention indexing

::: tileops.attention.FP8LightningIndexerFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
