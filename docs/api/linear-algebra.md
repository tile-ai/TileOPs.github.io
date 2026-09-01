# Linear Algebra Operators

Every op on this page is used the same way: construct it once, then call it. The
constructor takes what the kernel is compiled with; the call takes the tensors.
Both are documented under each op — `__init__` and `forward`, where `forward` is
what runs when you call `op(...)`.

## Dense GEMM

::: tileops.gemm.GemmFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
      merge_init_into_class: false
      show_signature_annotations: false

::: tileops.gemm.GemmFp8FwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
      merge_init_into_class: false
      show_signature_annotations: false

::: tileops.gemm.GemmW4A16FwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
      merge_init_into_class: false
      show_signature_annotations: false

## Batched GEMM

::: tileops.gemm.BmmFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
      merge_init_into_class: false
      show_signature_annotations: false

::: tileops.gemm.BmmFp8KNFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
      merge_init_into_class: false
      show_signature_annotations: false

::: tileops.gemm.BmmFp8NKFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
      merge_init_into_class: false
      show_signature_annotations: false

## Grouped GEMM

::: tileops.gemm.GroupedGemmFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
      merge_init_into_class: false
      show_signature_annotations: false
