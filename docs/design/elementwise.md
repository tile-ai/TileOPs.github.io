# Elementwise Operators

## Overview

TileOPs provides **66 elementwise operators** organized into three template families:

| Template | Description | Example ops |
|:---------|:-----------|:------------|
| **UnaryKernel / UnaryOp** | 1 input → 1 output | `relu`, `sigmoid`, `abs`, `exp`, ... |
| **BinaryKernel / BinaryOp** | 2 inputs → 1 output, N-dim broadcast | `add`, `mul`, `div`, ... |
| **FusedGatedKernel / FusedGatedOp** | Fused gate + activation | `silu_and_mul`, `gelu_and_mul`, ... |

All 55 concrete ops are registered as `torch.library.custom_op` at module load time, enabling full `torch.compile` support.

## Kernel Strategies

Each kernel uses one of three strategies, all operating without shared memory:

```
Global Memory → Register → Compute → Register → Global Memory
```

| Strategy | Description | Applicable to |
|:---------|:-----------|:-------------|
| **direct** | 1 element per thread, simplest codegen | Unary, Binary, FusedGated |
| **explicit_parallel** | N elements per thread via `T.Parallel(threads, npt)` | Unary, Binary, FusedGated |
| **register_copy** | Fragment load → compute → fragment store | Unary only |

!!! note
    Binary `register_copy` is **not supported** — it is incompatible with stride-based broadcast access patterns.

## Broadcast Coalescing

The Op layer (`BinaryOp`) uses `coalesce_broadcast_dims` to reduce an N-dimensional broadcast problem into the minimal number of effective dimensions before launching the kernel. This avoids unnecessary per-element stride computation on the GPU.

## FP8 Support

TileOPs supports `e4m3fn` and `e5m2` fp8 formats with an accumulation-in-fp16 strategy:

```
fp8 input → cast to fp16 → compute → cast back to fp8
```

Direct fp8 arithmetic loses too much precision for non-trivial ops (`sigmoid`, `exp`, etc.), so all computation is performed in fp16.

**Defaults for fp8:**

- `num_per_thread = 16` (1 byte × 16 = 128-bit memory alignment)
- Strategy: `explicit_parallel` (`register_copy` is unreliable for fp8)

**Saturation semantics** (matches NVIDIA spec):

| Format | Behavior |
|:-------|:---------|
| `e4m3fn` | No Inf/NaN representation; kernel uses `T.Cast` (saturating), clamps overflow to ±448.0 |
| `e5m2` | Has Inf/NaN; kernel outputs fp16 to preserve non-finite values, Op layer performs final non-saturating cast via PyTorch `.to()` |
