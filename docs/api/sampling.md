# Sampling Operators

Every op on this page is used the same way: construct it once, then call it. The
constructor takes what the kernel is compiled with; the call takes the tensors.
Both are documented under each op — `__init__` and `forward`, where `forward` is
what runs when you call `op(...)`.

The masks set every logit a filter drops to `-inf`, so a softmax over the result
renormalizes over what is kept. The samplers draw from a distribution, and take the
random seed and offset as tensors so a draw is reproducible.

## Logits masks

::: tileops.sampling.TopKMaskFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.sampling.MinPMaskFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.sampling.TopPMaskFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.sampling.TopKTopPMaskFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

## Sampling

::: tileops.sampling.SamplingFromProbsFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]

::: tileops.sampling.ChainSpeculativeSamplingFwdOp
    options:
      show_root_heading: true
      heading_level: 3
      members: ["__init__", "forward"]
