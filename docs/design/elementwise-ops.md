# Elementwise Operator Family

## 1. Overview

TileOPs elementwise operator family covers **72 ops** across 12 sub-categories, with fp8 support, torch.compile integration, kernel caching, and performance tuning.

### Scope (72 ops)

| Category                   | Ops                                                                                                             | Count |
| ----------------------------| -----------------------------------------------------------------------------------------------------------------| -------|
| **unary_math**             | exp, log, sqrt, rsqrt, abs, neg, reciprocal, sign, sin, cos, floor, ceil, round, trunc, erf, log1p, expm1       | 17    |
| **binary_arith**           | add, sub, mul, div, remainder, pow, floor_divide, lerp, maximum, minimum                                        | 10    |
| **activation**             | relu, gelu, silu, sigmoid, tanh, leaky_relu, elu, selu, hardswish, hardsigmoid, hardtanh, softplus, mish, prelu | 14    |
| **fused_gated_activation** | silu_and_mul (SwiGLU), gelu_and_mul, gelu_tanh_and_mul                                                          | 3     |
| **comparison**             | eq, ne, gt, lt, ge, le                                                                                          | 6     |
| **bitwise**                | bitwise_and, bitwise_or, bitwise_xor, bitwise_not                                                               | 4     |
| **logical**                | logical_not, logical_and, logical_or                                                                            | 3     |
| **special_elementwise**    | where, clamp, masked_fill, nan_to_num, isnan, isinf, isfinite                                                   | 7     |
| **rope**                   | neox, non_neox, llama31, yarn, longrope                                                                         | 5     |
| **dropout**                | dropout                                                                                                         | 1     |
| **alibi**                  | ALiBi position encoding                                                                                         | 1     |
| **sinusoidal**             | Sinusoidal position encoding                                                                                    | 1     |
| | | **72** |

### Supported Dtypes

fp32, bf16, fp16, fp8_e4m3fn, fp8_e5m2

---

## 2. Design Decisions

### 2.1 Kernel Template Strategy

Ops are grouped by input signature into template base classes. Each concrete op is a 3-5 line subclass overriding only `op_func`. Ops with user-configurable parameters or non-standard signatures are implemented independently.

**Template to Op Mapping:**

| Template | Signature | Ops | Count |
|----------|-----------|-----|-------|
| **UnaryKernel** | `op_func(x) → y` | unary_math (17); activation (9): relu, gelu, silu, sigmoid, tanh, hardswish, hardsigmoid, mish, selu; logical_not (1); bitwise_not (1); isnan, isinf, isfinite (3) | **31** |
| **BinaryKernel** | `op_func(a, b) → y`, with `output_dtype` and broadcast | binary_arith (10); comparison (6); logical_and, logical_or (2); bitwise_and, bitwise_or, bitwise_xor (3) | **21** |
| **FusedGatedKernel** | `activation_func(gate) * value`, input `(M, 2N)` split | silu_and_mul, gelu_and_mul, gelu_tanh_and_mul | **3** |
| **Independent** | Custom signatures, no shared template base class | leaky_relu, elu, hardtanh, softplus, prelu, where, clamp, masked_fill, nan_to_num, alibi, sinusoidal | **11** |
| | | **Total** | **66** |

> rope (5 ops) and dropout (1 op) are in separate files (`rope.py`, `dropout.py`) due to fundamentally different kernel patterns (complex rotary addressing, PRNG state management).

**Notes:**

- hardsigmoid、selu have fixed constants (not user-configurable), hardcoded in `op_func`, classified under UnaryKernel.
- softplus has user-configurable beta/threshold → independent implementation.
- Comparison ops output `int8` at the kernel level; the Op layer casts to `torch.bool`.

### 2.2 File Organization

```
tileops/kernels/
├── elementwise.py          # All template base classes + concrete kernel subclasses + independent kernels (~3,000 lines)
├── rope.py                 # 5 rope variants
├── dropout.py              # Dropout with PRNG

tileops/ops/
├── elementwise.py          # All template base Op classes + concrete op subclasses + custom_op registration (~1,800 lines)

tests/
├── ops/test_special_elementwise.py
├── test_elementwise_fp8.py
├── test_elementwise_compile.py
├── test_elementwise_independent_fp8.py
├── test_elementwise_strategy_bench.py
├── test_elementwise_caching_autotune.py

benchmarks/ops/
├── bench_unary_elementwise.py
├── bench_binary_elementwise.py
├── bench_independent_elementwise.py
├── bench_elementwise_fp8.py
```

Single `elementwise.py` for both kernels and ops. Tests split by concern (correctness, fp8, compile, strategy, caching). Benchmarks split by kernel template.

### 2.3 Re-export Strategy

- **`tileops/kernels/__init__.py`**: Re-exports only 3 base classes (`UnaryKernel`, `BinaryKernel`, `FusedGatedKernel`).
- **`tileops/ops/__init__.py`**: Re-exports only 3 base Op classes (`UnaryOp`, `BinaryOp`, `FusedGatedOp`).
- Concrete classes imported directly: `from tileops.kernels.elementwise import ReluKernel`.

### 2.4 Concrete Kernel Class per Op

Each op is an explicit class:

```python
class ReluKernel(UnaryKernel):
    @staticmethod
    def op_func(x):
        return T.max(x, 0)
```

Chosen over factory functions for: IDE navigation, `isinstance` checks, traceback clarity, and per-op `default_config` / `autotune_configs` overrides.

### 2.5 torch.compile Support: custom_op Registration

All 66 ops support `torch.compile` via `@torch.library.custom_op` registration.

**Registry mechanism**: `_OP_REGISTRY = weakref.WeakValueDictionary()` keyed by `id(instance)`. Each Op instance registers itself at construction time; the custom_op wrapper looks up the instance by key and delegates to `_eager_forward`.

**Factory functions** (7 total, registered at module load time):

| Factory | Ops | Signature |
|---------|-----|-----------|
| `_register_unary_custom_op` | 31 unary ops | `(x, instance_key) → Tensor` |
| `_register_binary_custom_op` | 21 binary ops | `(a, b, out_shape, instance_key) → Tensor` |
| `_register_fused_gated_custom_op` | 3 fused gated ops | `(x, M, N, instance_key) → Tensor` |
| `_register_prelu_custom_op` | prelu | `(x, weight, instance_key) → Tensor` |
| `_register_where_custom_op` | where | `(cond, x, y, instance_key) → Tensor` |
| `_register_masked_fill_custom_op` | masked_fill | `(x, mask, instance_key) → Tensor` |
| `_register_generative_custom_op` | alibi, sinusoidal | `(device_carrier, num_a, num_b, instance_key) → Tensor` |

Each Op class's `forward` dispatches through `_wrapped(...)` when torch.compile is active, falling back to `_eager_forward` otherwise.

---

## 3. Kernel Architecture

### 3.1 Shape Convention

**Unary ops**: Arbitrary shape, flattened to 1D `(N_total,)` by the Op layer before passing to kernel.

**Binary ops**: Stride-based offset with N-dim broadcast (equivalent to PyTorch TensorIterator semantics).

Op layer processing:
1. `torch.broadcast_shapes(a.shape, b.shape)` → output shape
2. Align dimensions (left-pad with 1), compute broadcast strides (stride=0 for broadcast dims)
3. **Dimension coalescing** (`coalesce_broadcast_dims`): merge adjacent dims with compatible broadcast patterns to minimize divmod count
4. Both inputs `.contiguous().view(-1)` flattened to 1D; `(coalesced_shape, a_strides, b_strides)` passed as compile-time constants
5. Kernel computes output via flat index (contiguous), maps to input offsets through divmod chain

**FusedGated ops**: Input `(M, 2N)` → 2D grid, no broadcast.

### 3.2 Memory Access Pattern: No Shared Memory

Elementwise ops are pure memory-bandwidth bound (arithmetic intensity ≈ 0). The optimal path is:

```
Global Memory ──direct read──▶ Register ──Compute──▶ Register ──direct write──▶ Global Memory
```

- No shared memory allocation or budget constraints
- `T.Parallel` maps threads + 128-bit vectorized coalesced access (`uint4` loads/stores)
- `LegalizeSafeMemoryAccess` compiler pass auto-inserts boundary guards

### 3.3 Three Kernel Strategies

#### Strategy 1: `direct` — Compiler-optimized

Each thread processes 1 element. TileLang may auto-vectorize. Least code, but compiler behavior not fully controllable.

```python
def _make_unary_direct(N, dtype, op_func, output_dtype=None, threads=256):
    @T.prim_func
    def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), out_dtype)):
        with T.Kernel(T.ceildiv(N, threads), threads=threads) as bx:
            for i in T.Parallel(threads):
                y[bx * threads + i] = op_func(x[bx * threads + i])
    return main
```

#### Strategy 2: `explicit_parallel` — Explicit vectorization control

Each thread processes `num_per_thread` elements. `T.Parallel(threads, num_per_thread)` explicitly guides the compiler for vectorized access.

```python
def _make_unary_explicit(N, dtype, op_func, output_dtype=None, threads=256, num_per_thread=8):
    block_size = threads * num_per_thread
    @T.prim_func
    def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), out_dtype)):
        with T.Kernel(T.ceildiv(N, block_size), threads=threads) as bx:
            for i, j in T.Parallel(threads, num_per_thread):
                idx = (bx * threads + i) * num_per_thread + j
                y[idx] = op_func(x[idx])
    return main
```

#### Strategy 3: `register_copy` — Explicit register staging

Explicit `T.alloc_fragment` + `T.copy` for global↔register transfers. `T.copy` generates vectorized load/store instructions.

```python
def _make_unary_regcopy(N, dtype, op_func, output_dtype=None, threads=256, num_per_thread=8):
    block_size = threads * num_per_thread
    @T.prim_func
    def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), out_dtype)):
        with T.Kernel(T.ceildiv(N, block_size), threads=threads) as bx:
            x_reg = T.alloc_fragment((block_size,), dtype)
            y_reg = T.alloc_fragment((block_size,), out_dtype)
            T.copy(x[bx * block_size:(bx + 1) * block_size], x_reg)
            for i, j in T.Parallel(threads, num_per_thread):
                y_reg[i * num_per_thread + j] = op_func(x_reg[i * num_per_thread + j])
            T.copy(y_reg, y[bx * block_size:(bx + 1) * block_size])
    return main
```

#### Binary Kernel: Stride-Based Offset

Binary ops use 1D output flat index + stride-based offset for N-dim broadcast. All shape/stride values are compile-time constants; the divmod chain is unrolled at compilation.

All three strategies (direct, explicit_parallel, register_copy) are supported. When `register_copy` is requested on broadcast inputs (ndim > 1), it silently downgrades to `explicit_parallel` because `T.copy` requires contiguous slices.

#### Strategy Defaults (H200 benchmarks)

| Kernel Type | Default Strategy | Rationale |
|-------------|-----------------|-----------|
| **UnaryKernel** | `register_copy` | H200: register_copy wins for fp16/bf16 across all tested shapes |
| **BinaryKernel** | `explicit_parallel` | Broadcast inputs incompatible with register_copy; explicit_parallel is most general |
| **FusedGatedKernel** | `explicit_parallel` | H200: ~2× faster than direct (e.g. silu_and_mul: 3.04 TB/s vs 1.50 TB/s) |
| **fp8 (all types)** | `explicit_parallel` | register_copy unreliable for 8-bit fragments |

#### Config Defaults

Strategy-aware `num_per_thread` defaults (H200 benchmarks):

| Strategy | fp32 | fp16/bf16 | fp8 |
|----------|------|-----------|-----|
| `explicit_parallel` | 4 | 4 | 16 |
| `register_copy` | 4 | 8 | — |
| `direct` | — | — | — |

Rationale:
- fp32 × 4B × 4 = 16B = 128-bit `uint4` alignment
- fp16/bf16 under `explicit_parallel`: npt=4 yields 42% bandwidth gain over npt=8
- fp16/bf16 under `register_copy`: npt=8 × 2B = 16B = 128-bit vectorized loads
- fp8: npt=16 × 1B = 16B = 128-bit alignment

`threads` fixed at 256 for all strategies. Elementwise ops have low register pressure; 256 threads achieves good occupancy on SM80+.

### 3.4 Template Base Classes

#### UnaryKernel

```python
class UnaryKernel(Kernel):
    supported_archs = [80, 86, 89, 90]
    STRATEGIES = ["direct", "explicit_parallel", "register_copy"]
    DEFAULT_STRATEGY = "register_copy"
    OUTPUT_DTYPE = None        # subclass override (e.g. torch.bool for predicates)
    SUPPORTED_DTYPES = None    # subclass override to restrict input dtypes

    @staticmethod
    def op_func(x):
        raise NotImplementedError

    def __init__(self, N_total, dtype, strategy=None, config=None, tune=False):
        # dtype validation
        # fp8: override strategy to explicit_parallel
        # _build_kernel → _make_unary_{direct,explicit,regcopy}
        # init_config → cache _compiled_fn

    def forward(self, x):
        return self._compiled_fn(x)
```

Key implementation details:
- `_build_kernel` dispatches to the appropriate `_make_unary_*` factory based on strategy
- `init_config` caches `_compiled_fn` by pre-applying config params (threads, npt) to the kernel builder
- fp8 handling: `_wrap_fp8_accumulation` wraps `op_func` with cast-in/cast-out; `_fp8_output_dtype` signals the Op layer to apply post-cast
- Autotune override catches serialization failures (op_func closures aren't pickleable) and falls back to default_config

#### BinaryKernel

```python
class BinaryKernel(Kernel):
    supported_archs = [80, 86, 89, 90]
    STRATEGIES = ["direct", "explicit_parallel", "register_copy"]
    DEFAULT_STRATEGY = "explicit_parallel"
    OUTPUT_DTYPE = None        # "int8" for comparison/logical ops
    SUPPORTED_DTYPES = None

    @staticmethod
    def op_func(a, b):
        raise NotImplementedError

    def __init__(self, N_total, dtype, coalesced_shape, a_strides, b_strides,
                 a_numel, b_numel, strategy=None, config=None, tune=False):
        # dtype validation
        # register_copy auto-downgrade on broadcast inputs
        # _build_kernel → _make_binary_{direct,explicit,register_copy}
        # init_config → cache _compiled_fn

    def forward(self, a, b):
        return self._compiled_fn(a, b)
```

- Comparison ops use `OUTPUT_DTYPE = "int8"` at kernel level; Op layer (`_BoolOutputBinaryOp`) casts to `torch.bool`
- `register_copy` silently downgrades to `explicit_parallel` when any input has broadcast dimensions (ndim > 1)

#### FusedGatedKernel

```python
class FusedGatedKernel(Kernel):
    supported_archs = [80, 86, 89, 90]
    STRATEGIES = ["direct", "explicit_parallel"]
    DEFAULT_STRATEGY = "explicit_parallel"
    SUPPORTED_DTYPES = None

    @staticmethod
    def activation_func(x):
        raise NotImplementedError

    def __init__(self, M, N, dtype, strategy=None, config=None, tune=False):
        # _build_kernel → _make_fused_gated_{direct,explicit}
        # init_config → cache _compiled_fn

    def forward(self, x):
        return self._compiled_fn(x)
```

- Input: single `(M, 2N)` tensor (gate + value packed); output: `(M, N)`
- No `register_copy` — FusedGated has a 2D grid pattern incompatible with 1D fragment copy
- Single tensor interface matches vLLM/FlashInfer/xformers convention

#### Independent Ops (11 ops)

Each has its own builder function, `__init__`, and `forward`. Shared patterns:
- `_make_*_kernel` with `is_fp8` branching: fp8 uses explicit_parallel with fp16 accumulation; non-fp8 uses register_copy
- `_compiled_fn` caching in `init_config`
- `_fp8_output_dtype` for Op-layer post-cast

| Op | Extra Parameters | Builder |
|----|-----------------|---------|
| leaky_relu | negative_slope (scalar) | `_make_leaky_relu_kernel` |
| elu | alpha (scalar) | `_make_elu_kernel` |
| hardtanh | min_val, max_val (scalars) | `_make_hardtanh_kernel` |
| softplus | beta, threshold (scalars) | `_make_softplus_kernel` |
| prelu | weight (tensor) | `_make_prelu_kernel` |
| where | — | `_make_where_kernel` |
| clamp | min_val, max_val (scalars) | `_make_clamp_kernel` |
| masked_fill | fill_value (scalar) | `_make_masked_fill_kernel` |
| nan_to_num | nan, posinf, neginf (scalars) | `_make_nan_to_num_kernel` |
| alibi | — | `_make_alibi_kernel` |
| sinusoidal | — | `_make_sinusoidal_kernel` |

---

## 4. fp8 Support

### 4.1 Accumulation Strategy

fp8 dtypes lack precision for intermediate computation. All fp8 kernels use fp16 accumulation:

| Input dtype | Accumulation | Output | Post-cast (Op layer) |
|-------------|-------------|--------|---------------------|
| **e4m3fn** | fp16 | e4m3fn (saturating `T.Cast`) | None — kernel output is already e4m3fn |
| **e5m2** | fp16 | fp16 (no cast in kernel) | Non-saturating `.to(e5m2)` preserving Inf/NaN |

The distinction: e4m3fn has no Inf representation, so saturating cast is correct. e5m2 supports Inf/NaN, so the final cast must be non-saturating (PyTorch `.to()` semantics).

### 4.2 Implementation

**Shared helper** `_wrap_fp8_accumulation(base_op, dtype, dtype_str, arity)`:
- Wraps any `op_func` with fp8 cast-in/cast-out logic
- Applied uniformly in `_get_effective_op_func` for all three template kernels
- Non-fp8 dtypes pass through unchanged

**Scalar clamping** `_clamp_to_dtype_range(value, dtype)`:
- Prevents TVM FloatImm range-check failures for scalar literals exceeding fp8 range (e.g., `fill_value=1e4` into e4m3fn whose max is 448)
- NaN passed through; Inf mapped to dtype max/min

**Config for fp8**:
- `npt=16` (1B × 16 = 128-bit alignment)
- Strategy forced to `explicit_parallel` (register_copy unreliable for 8-bit fragments)
- Autotune grid: `threads_opts=[128,256,512]`, `npt_opts=[16,32]`

---

## 5. Op Layer Architecture

### 5.1 Broadcast Utility: `coalesce_broadcast_dims`

The core utility for binary ops. Converts arbitrary N-dim broadcast into minimal `(coalesced_shape, a_strides, b_strides)`.

Algorithm:
1. `torch.broadcast_shapes` → output shape
2. Right-align, left-pad with 1
3. Compute C-contiguous strides (stride=0 for broadcast dims)
4. Merge adjacent dims where all operands have compatible stride patterns
5. Remove trivial size-1 groups

**Coalescing effect:**

| Scenario | shapes | Coalesced ndim | Divmod count |
|----------|--------|---------------|-------------|
| same-shape | `(B,S,D) + (B,S,D)` | 1 | 0 |
| bias add | `(B,S,D) + (1,1,D)` | 2 | 1 |
| scaling | `(B,S,D) + (B,S,1)` | 2 | 1 |
| attn mask | `(B,H,S,S) + (1,1,S,S)` | 2 | 1 |
| interleaved | `(B,H,S,S) + (B,1,1,S)` | 3 | 2 |
| outer product | `(M,1) + (1,N)` | 2 | 1 |

### 5.2 Template Op Classes

**UnaryOp**:
```python
class UnaryOp(Op):
    def forward(self, x):
        orig_shape = x.shape
        x = x.contiguous().reshape(-1)     # flatten to 1D
        y = self.kernel(x)                  # kernel handles boundary
        y = y.reshape(orig_shape)
        return _apply_fp8_post_cast(y, self.kernel)  # e5m2 non-saturating cast
```

**BinaryOp**:
```python
class BinaryOp(Op):
    def __init__(self, a_shape, b_shape, dtype, ...):
        out_shape, coalesced_shape, a_strides, b_strides = coalesce_broadcast_dims(a_shape, b_shape)
        self.kernel = self.kernel_cls(N_total, dtype, coalesced_shape, a_strides, b_strides, ...)

    def forward(self, a, b):
        a_flat = a.contiguous().view(-1)
        b_flat = b.contiguous().view(-1)
        y = self.kernel(a_flat, b_flat)
        return y.reshape(self.out_shape)
```

**FusedGatedOp**: Input `(M, 2N)` → kernel → output `(M, N)`. No reshape needed.

All Op classes include validation: CUDA device check, dtype matching, numel matching.

### 5.3 torch.compile Dispatch

Each Op's `forward` checks for `_wrapped`:
- If `_wrapped` is set: dispatch through custom_op (supports torch.compile tracing)
- Otherwise: call `_eager_forward` directly

Instance-based registry (`_OP_REGISTRY`) enables the custom_op wrapper to look up the correct Op instance at runtime.

---

## 6. Kernel Caching

Each kernel instance pre-compiles its kernel function in `init_config`:

```python
def init_config(self, config=None, tune=False):
    super().init_config(config, tune)
    cfg = self.config
    self._compiled_fn = self.kernel(cfg["threads"], cfg["num_per_thread"])
```

This avoids JIT lookup overhead on every `forward()` call. The compiled function is cached per-instance, not globally.

Autotune serialization: `op_func`/`activation_func` closures aren't pickleable. The autotune override catches serialization failures and falls back to `default_config`.

