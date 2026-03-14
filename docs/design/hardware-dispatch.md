
# Hardware-Aware Dispatch

## Context

Different GPU architectures offer distinct instruction sets and performance characteristics:

| Architecture | SM Version | Key Features |
|:-------------|:-----------|:-------------|
| Ampere | SM 80/86 | Tensor Cores (FP16/BF16/TF32/INT8), async copy |
| Hopper | SM 90 | WGMMA, TMA, warp specialization, FP8 |

A single kernel implementation cannot optimally serve all architectures.

## Decision

Each Op declares a `default_kernel_map` that maps SM architecture identifiers to Kernel classes:

```python
class MultiHeadAttentionFwdOp(Op):
    @property
    def default_kernel_map(self):
        return {
            "ampere": FlashAttnV2FwdKernel,
            "hopper": FlashAttnV3FwdKernel,
        }
```

At runtime, `dispatch_kernel()` queries the GPU's SM version via `get_sm_version()` and selects the matching kernel.

## SM Version Detection

```python
# tileops/utils/
def get_sm_version() -> int:
    """Returns SM version of the current CUDA device (e.g., 80, 86, 90)."""
```

## Consequences

- Ops automatically use the best kernel for the current hardware
- New architectures can be added by extending the kernel map
- Users can override the kernel map for custom dispatch logic
- Fallback behavior when no architecture-specific kernel exists
