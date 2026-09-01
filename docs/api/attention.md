# Attention Operators

Every op on this page is used the same way: construct it once, then call it. The
constructor takes what the kernel is compiled with; the call takes the tensors.
Both are documented under each op — `__init__` and `forward`, where `forward` is
what runs when you call `op(...)`.

## Multi-head attention

::: tileops.attention.MultiHeadAttentionFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.attention.MultiHeadAttentionBwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.attention.MultiHeadAttentionDecodeWithKVCacheFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.attention.MultiHeadAttentionDecodePagedWithKVCacheFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Grouped-query attention

::: tileops.attention.GroupedQueryAttentionFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.attention.GroupedQueryAttentionBwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.attention.GroupedQueryAttentionDenseFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.attention.GroupedQueryAttentionPrefillFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.attention.GroupedQueryAttentionPrefillVarlenFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.attention.GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.attention.GroupedQueryAttentionDecodeWithKVCacheFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.attention.GroupedQueryAttentionDecodePagedWithKVCacheFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.attention.GroupedQueryAttentionSlidingWindowFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.attention.GroupedQueryAttentionSlidingWindowVarlenFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Multi-head latent attention

::: tileops.attention.MultiHeadLatentAttentionDecodeWithKVCacheFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Native sparse attention

::: tileops.attention.NSACmpFwdVarlenOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.attention.NSATopkVarlenOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.attention.NSAFwdVarlenOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## DeepSeek sparse attention

::: tileops.attention.DeepSeekSparseAttentionDecodeWithKVCacheFwdOp
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
