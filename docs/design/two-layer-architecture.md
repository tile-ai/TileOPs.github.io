
# Two-Layer Architecture

## Context

TileOPs needs to support multiple GPU architectures (Ampere, Hopper) with architecture-specific optimizations while presenting a clean, stable API to users.

## Decision

We separate the system into two layers:

### L1 — Kernel Layer

- **Written in TileLang** — GPU-native kernel implementations
- **Hardware-specific** — Separate implementations per SM architecture when beneficial
- **Configuration-driven** — Block sizes, pipeline stages, and other tuning parameters are externalized
- **Autotunable** — Each kernel declares `autotune_configs` for performance search

```
tileops/kernels/
├── flash_attn/          # Flash Attention v2 (Ampere), v3 (Hopper)
├── flash_decode/        # Paged KV-cache decode kernels
├── gemm/                # Matrix multiplication
├── norm/                # LayerNorm, RmsNorm, BatchNorm, etc.
├── reduction/           # Softmax, reduce, cumulative, argreduce
├── elementwise.py       # Unary + binary elementwise ops
└── kernel.py            # Kernel base class
```

### L2 — Op Layer

- **Pure Python** — Stateless dispatchers
- **Framework-compatible** — Works with CUDA-Graph and `torch.compile`
- **Dtype-validated** — Explicit input/output dtype contracts
- **Architecture-aware** — Queries SM version at runtime, selects appropriate kernel

```
tileops/ops/
├── attention/           # MHA, GQA, MLA, NSA, Linear Attention
├── gemm/                # GEMM, Grouped GEMM
├── norm/                # All normalization ops
├── reduction/           # Softmax, reduce, cumulative
├── elementwise/         # Unary, binary, fused
└── op.py                # Op base class
```

## Dispatch Flow

```
User calls Op.forward(inputs)
  → Op validates dtypes
  → Op.dispatch_kernel() selects kernel by SM version
  → Kernel.forward() launches TileLang program on GPU
  → Results returned to user
```

## Consequences

**Benefits:**
- Clean separation of concerns — hardware complexity stays in L1
- Multiple kernel implementations per op (e.g., Flash Attention v2 vs v3)
- User code is architecture-agnostic
- Autotuning at kernel level without touching op interface

**Trade-offs:**
- Two files to create per operator (kernel + op)
- Kernel map maintenance when adding new architectures
