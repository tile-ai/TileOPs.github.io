# kernel 内的时间线追踪 { #in-kernel-timeline-trace }

[`tileops.trace`](../api/trace.md) 是一个在 kernel 内部记录时间线的追踪工具，用于诊断 kernel 的性能。它在 kernel 内为每个 CTA 记录时间戳，并渲染成一条可以滚动查看的时间线，显示执行区间、空隙，以及 producer 与 consumer 之间的重叠。这些信息是 `ncu` 这类以 kernel 为单位的 profiler 看不到的。warp specialization 的 kernel 最能用上它：这类 kernel 的关键就是让 producer（TMA）与 consumer（WGMMA）两个 warpgroup 的执行相互重叠。

## 工作方式 { #how-it-works }

1. 在 kernel 代码中用标记（`trace.range`、`trace.group` 等）标注要记录的区间。
1. 标记**总是以占位的形式生成**。构建时，kernel 要么被 **lower**：标记变成真正调用 `clock64()` 记录时间的代码，并在输出末尾多出一个 `slots`；要么被 **strip**：标记变成空操作，生成的 CUDA 与未插桩的版本完全相同。
1. 运行时，`trace.run` 执行 kernel，解码 `slots` 缓冲区，并写出一个自包含的 Plotly HTML 时间线。
1. 由进程内的开关 `trace.enable()` 决定 lower 还是 strip，因此关闭时追踪**没有任何开销**，标记可以留在生产代码中。

时间戳来自 `clock64()`，即每个 SM 的周期计数器。

## 编写带追踪的 kernel { #write-a-traced-kernel }

下面是一个完整的、接入了追踪的 warp specialization GEMM（示意用，单缓冲）。编号 `(1)`–`(7)` 的标记是仅有的与追踪相关的代码，各自的说明和 API 文档链接见代码下方。生产环境中的多 stage 版本见 [`src/tileops/kernels/gemm/dense.py`](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/kernels/gemm/dense.py)。

```{ .python .annotate }
import functools
import tilelang
import tilelang.language as T
from tileops.trace import trace  # (1)!


@functools.lru_cache(maxsize=32)
def build_gemm(m, n, k, dtype="float16", traced=False):
    @tilelang.jit(out_idx=trace.out_idx(1, traced))  # (2)!
    def factory(block_m=128, block_n=128, block_k=64):
        @T.prim_func
        def main(a: T.Tensor((m, k), dtype), b: T.Tensor((n, k), dtype),
                 c: T.Tensor((m, n), dtype)):
            with T.Kernel(T.ceildiv(n, block_n), T.ceildiv(m, block_m),
                          threads=256) as (bx, by):
                a_smem = T.alloc_shared((block_m, block_k), dtype)
                b_smem = T.alloc_shared((block_n, block_k), dtype)
                c_local = T.alloc_fragment((block_m, block_n), "float")
                full = T.alloc_barrier(128)
                tx = T.get_thread_binding()

                if tx < 128:
                    with trace.group("producer", lead=0):  # (3)!
                        for ki in T.serial(T.ceildiv(k, block_k)):
                            with trace.range("tma", lane="tma"):  # (4)!
                                T.tma_copy(a[by * block_m, ki * block_k], a_smem, barrier=full)
                                T.tma_copy(b[bx * block_n, ki * block_k], b_smem, barrier=full)
                            with trace.range("arrive", lane="barrier"):
                                T.barrier_arrive(full)
                else:
                    with trace.group("consumer", lead=128):
                        T.clear(c_local)
                        for ki in T.serial(T.ceildiv(k, block_k)):
                            with trace.range("wait", lane="barrier"):
                                T.barrier_wait(full, ki % 2)
                            with trace.range("mma", lane="wgmma"):  # (5)!
                                T.wgmma_gemm(a_smem, b_smem, c_local, transpose_B=True)
                        with trace.range("epilogue"):
                            T.copy(c_local, c[by * block_m, bx * block_n])

                trace.dag("arrive", "wait")  # (6)!
        return trace.finalize(main, traced=traced, max_events=1024)  # (7)!
    return factory
```

1. 导入 trace 命名空间。下面的每个调用都是这个 `trace` 对象上的方法，完整说明见 [API 参考](../api/trace.md)。
1. [`trace.out_idx(n_outputs, traced)`](../api/trace.md#tileops.trace.api._Trace.out_idx) 给出 `@tilelang.jit` 的 `out_idx`。只有 `traced` 时它才多出一个位置给末尾的 `slots` 输出，因此同一个构建函数在开启和关闭追踪时都能使用。
1. [`trace.group(name, lead)`](../api/trace.md#tileops.trace.api._Trace.group) 声明由哪个 warpgroup 记录。`lead` 是被选出的写入线程（`tx == lead`）：计算仍在所有线程上执行，只有时间戳由 `lead` 写入。
1. [`trace.range(name, lane)`](../api/trace.md#tileops.trace.api._Trace.range) 是一个 `with` 块，从进入计时到退出，在子行 `lane` 上画成一段条形。控制流放不进 `with` 时，用 [`trace.range_start`](../api/trace.md#tileops.trace.api._Trace.range_start) 与 [`trace.range_end`](../api/trace.md#tileops.trace.api._Trace.range_end)；只需要一个零宽度的时间点时，用 [`trace.record`](../api/trace.md#tileops.trace.api._Trace.record)。
1. lane 的名字（`"tma"`、`"barrier"`、`"wgmma"`，以及默认的 `"main"`）就是时间线上的各行。
1. [`trace.dag(src, dst)`](../api/trace.md#tileops.trace.api._Trace.dag) 声明从一个命名区间指向另一个区间的依赖箭头（`arrive` → `wait`），每出现一次画一条。
1. [`trace.finalize(func, traced, max_events)`](../api/trace.md#tileops.trace.api._Trace.finalize) 在 `traced` 时 lower 标记并加上 `slots` 输出，否则把标记 strip 成零开销。`traced` **必须是构建函数缓存键的一部分**，这样同一形状下带追踪和不带追踪的两次构建才不会冲突。

## 运行带追踪的 kernel { #running-a-traced-kernel }

调用构建函数时传入 `traced=trace.enabled`，再把编译好的 kernel 交给 `trace.run`。同一个 `forward` 在两种模式下都能使用：追踪关闭时原样返回输出；追踪开启时写出时间线，并且只返回真正的输出。因此调用方不需要自己按开关分支。

```{ .python .annotate }
def forward(self, a, b):
    compiled = build_gemm(self.m, self.n, self.k, self.dtype_str,
                          traced=trace.enabled)(**self.config)  # (1)!
    return trace.run(compiled, (a, b), stem="gemm_128x256x512")  # (2)!
```

1. 按开关构建对应的 kernel：[`trace.enabled`](../api/trace.md#tileops.trace.api._Trace.enabled) 决定选用带追踪的版本还是 strip 后的版本，它也是标记 `(7)` 所说的缓存键的一部分。
1. [`trace.run(compiled, inputs, stem=...)`](../api/trace.md#tileops.trace.api._Trace.run) 运行 kernel。带追踪时，它把末尾的 `slots` 拆出来解码，写出 `debug/<stem>.html`，每次调用都生成一个不会重名的新文件。它内部由 [`decode`](../api/trace.md#tileops.trace.api._Trace.decode) 和 [`dump`](../api/trace.md#tileops.trace.api._Trace.dump) 组成，两者也可以直接调用。

## 开启追踪 { #enabling-tracing }

追踪默认关闭。在程序启动时打开一次进程内的开关，之后照常运行：

```{ .python .annotate }
from tileops.trace import trace

trace.enable()  # (1)!
c = op.forward(a, b)  # (2)!
```

1. [`trace.enable(output="debug")`](../api/trace.md#tileops.trace.api._Trace.enable) 打开追踪，并指定输出目录，默认是 `debug/`（已被 gitignore）。这个开关只在当前进程内有效：不读取环境变量，也不对 `tilelang` 做 monkeypatch。相关的还有 [`trace.disable()`](../api/trace.md#tileops.trace.api._Trace.disable)、[`trace.enabled`](../api/trace.md#tileops.trace.api._Trace.enabled) 与 [`trace.output`](../api/trace.md#tileops.trace.api._Trace.output)。
1. 此后运行的每个带追踪的 kernel 都会写出 `debug/<stem>.html`。

在 pytest 中，`--trace-kernel` 会在任何 kernel 构建之前，由 `pytest_configure` 调用 `trace.enable()`：

```bash
pytest tests/ops/test_gemm.py --trace-kernel
```

## 阅读时间线 { #reading-the-timeline }

<iframe src="../../../performance-guides/gemm-trace.html" title="GEMM 时间线" width="100%" height="540"
        style="border:1px solid var(--md-default-fg-color--lightest);border-radius:4px;"></iframe>

- **顶部的 CTA 标签页**：每个 CTA（block）一条时间线。
- **lane**（各行）来自代码中 `group` 与 `lane` 的名字，例如 `producer / tma`、`consumer / wgmma`。每个 `range` 是一段条形，鼠标悬停可以看到它的名字与周期区间。
- **横轴**是 SM 的原始周期数（`clock64()`），每个 CTA 从零开始。
- **箭头**是代码中声明的 `dag` 边（例如 producer 的 `arrive` → consumer 的 `wait`），每出现一次画一条。从箭头可以读出交接的延迟，以及 consumer 是否在空等。
- 缩放与平移只在水平方向进行。

需要留意的现象：

- `wgmma` 这一行上有空隙：consumer 停下来等待 TMA 加载。
- 相邻迭代之间的 `dag` 箭头没有重叠：没有形成流水。
- 某一行明显比其他行长：负载不均衡。
