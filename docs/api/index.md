
# API Reference

TileOPs exposes **70+ operators** through a consistent two-layer API:

- **Op** (L2) — User-facing, stateless dispatchers with dtype validation and kernel selection
- **Kernel** (L1) — TileLang GPU implementations with hardware-specific optimizations

## Base Classes

### `Op`

All operators inherit from `tileops.ops.op.Op`:

```python
from tileops.ops import GemmOp

op = GemmOp(dtype=torch.float16)
result = op.forward(A, B)
```

Key attributes:
- `kernel` — Active kernel instance
- `kernel_map` — Available kernel implementations (keyed by architecture)
- `dtype` — Data type for computation
- `device` — Target device (default: `'cuda'`)

Key methods:
- `forward(*args, **kwargs)` — Execute the operation
- `autotune()` — Run autotuning to find optimal kernel configuration
- `dispatch_kernel(kernel_map)` — Select kernel based on hardware

### `Kernel`

All kernels inherit from `tileops.kernels.kernel.Kernel`:

Key attributes:
- `config` — Kernel configuration parameters (block sizes, pipeline stages, etc.)
- `autotune_configs` — List of configurations to search during autotuning
- `supported_archs` — Compatible SM architectures (e.g., `[80, 86, 90]`)

Key methods:
- `forward(*args, **kwargs)` — Execute the kernel
- `autotune(warmup=10, rep=10)` — Profile configurations and select the fastest
- `init_config(config=None, tune=False)` — Initialize or autotune kernel config

## Op Categories

Browse operators by category:
