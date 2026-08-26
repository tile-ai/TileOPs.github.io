# Attention Operators

Every op on this page is used the same way: construct it once, then call it. The
constructor takes what the kernel is compiled with; the call takes the tensors.
Both are documented under each op — `__init__` and `forward`, where `forward` is
what runs when you call `op(...)`.

## Multi-head attention

::: tileops.ops.attention.mha.MultiHeadAttentionFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.attention.mha.MultiHeadAttentionBwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.attention.mha.MultiHeadAttentionDecodeWithKVCacheFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.attention.mha.MultiHeadAttentionDecodePagedWithKVCacheFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Grouped-query attention

::: tileops.ops.attention.gqa.GroupedQueryAttentionFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.attention.gqa.GroupedQueryAttentionBwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.attention.gqa.GroupedQueryAttentionDenseFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.attention.gqa.GroupedQueryAttentionPrefillFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.attention.gqa.GroupedQueryAttentionPrefillVarlenFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.attention.gqa.GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.attention.gqa.GroupedQueryAttentionDecodeWithKVCacheFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.attention.gqa.GroupedQueryAttentionDecodePagedWithKVCacheFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.attention.gqa.GroupedQueryAttentionSlidingWindowFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.attention.gqa.GroupedQueryAttentionSlidingWindowVarlenFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Multi-head latent attention

::: tileops.ops.attention.deepseek_mla.MultiHeadLatentAttentionDecodeWithKVCacheFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Native sparse attention

::: tileops.ops.attention.deepseek_nsa.NSACmpFwdVarlenOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.attention.deepseek_nsa.NSATopkVarlenOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.ops.attention.deepseek_nsa.NSAFwdVarlenOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## DeepSeek sparse attention

::: tileops.ops.attention.deepseek_dsa.DeepSeekSparseAttentionDecodeWithKVCacheFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Attention indexing

::: tileops.ops.fp8_lightning_indexer.FP8LightningIndexerFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
