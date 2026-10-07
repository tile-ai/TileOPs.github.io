# benchmark 的计时方法

nightly benchmark 为每个 op 的每个 workload 各测一行，报出的 `device_busy_ms` 是这次调用产生的全部 kernel 在设备上执行区间的并集。CUPTI 记录每个 kernel 的执行起止时间，并按 external correlation id 把它归到某一次迭代。每次迭代之前清空 L2，预热 25 ms、测量 100 ms，结果取中位数。

**因此表中的每个数都是设备执行 kernel 的时间，不含 CPU 发起调用的开销，也不含 kernel 之间的空隙。阅读表格只需要知道这一点。**{ .keystone }

其余各节供需要时查阅：

- [一次测量的流程](#how-it-runs)：伪代码，以及校准、迭代次数、清空 L2、归属与失败即停五项安排。
- [被测量的量](#what-is-measured)：`device_busy_ms` 的定义，以及 kernel 之间的空隙不计入的原因。
- [为什么不是墙钟时间](#why-not-wall-clock)：在 decode 尺度上，CUDA events 测不出小 kernel 的执行时间。
- [什么时候要改测法](#when-to-change)：只在自行编写 benchmark 时用到，包括这套测法测不到的几种情形。

下文的数字都在 H200 上实测，镜像为 `tileops-runner:cu132-torch2.13`。

## 一次测量的流程 {#how-it-runs}

```python
from benchmarks.timing import bench_kernel

samples = bench_kernel(op, args=(x, weight))   # 每次迭代一个 Sample
```

编写 benchmark 时一般不直接调用 `bench_kernel`，而是通过 `bench.Runner(op, case).compare()` 计时；这一层在样本上取中位数，并计算派生列，写法见[编写 benchmark](user-guide/benchmark/writing.md)。

`bench_kernel` 内部分为采集、归属与计量三段：

```python
# 采集：每次调用在自己的迭代号下执行
with _phase_session():                          # kernel + 拷贝 + 映射 + launch 类 API
    for i in range(n_repeat):
        with _labelled(_PREPARE_ID):            # 准备工作用专用 id 标记
            prepare_one(i)
        with _labelled(i):                      # push i ... pop
            run_one(i)                          # 被测调用
        torch.cuda.synchronize()                # 等它排空，下次迭代不与它重叠
    kernels, iteration_of = _flush()            # 循环结束后取一次记录
    dropped = _read_dropped() if _drop_counter_is_live() else None

# 归属：kernel 携带的 correlation id 决定它属于哪次迭代
for kernel in kernels:
    i = iteration_of.get(kernel["correlation_id"])
    if i == _PREPARE_ID:
        continue                                # 准备工作不属于被计时的运算
    if i is None or not 0 <= i < n_repeat:
        orphans.append(kernel)                  # 本 phase 没标记过这个 id
    else:
        claimed[i].append(kernel)

# 计量：每次迭代一个样本
for i in range(n_repeat):
    busy.append(union(claimed[i]))              # 被测量的量，见下一节
    latency.append(max(end) - min(start))
    n_kernels.append(len(claimed[i]))
```

五项安排及其原因如下：

1. **校准。** 先运行 3 次，估计单次调用的耗时。
2. **换算迭代次数。** 用预热 25 ms、测量 100 ms 的预算除以单次耗时，结果限制在 `[10, 200]` 之内。耗时短的 op 因此采样更多，耗时长的 op 不必运行满 200 次。
3. **每次迭代之前清空 L2，并等待设备排空。** 不清空 L2 时，第一次迭代从 HBM 读取，之后的迭代都从 L2 读取，中位数反映的是缓存全部命中的最好情况。等待排空使上一次迭代不与这一次重叠。实现需要恢复被改写的输入时，恢复在它的 `reset` 中完成；`reset` 在清空 L2 之前执行，写入的数据因此不会留在 L2 中。
4. **采集与归属。** 每次迭代把迭代号标记为 CUPTI 的 external correlation id，区间内发出的每次 launch 都带上这个 id；kernel 记录中的 correlation id 再经这层映射对应回迭代号。**归属不依据时间戳**：一个 kernel 属于哪次迭代写在记录里，与它何时执行无关。因此比主机开销还短的 kernel 同样能被可靠归属，一次调用的 kernel 数在迭代之间变化时也能测出。
5. **失败即停。** 三种归属失败各报一种错误，不产出数字。

| 情形 | 报什么 | 含义 |
| --- | --- | --- |
| CUPTI 丢了记录 | `_CUPTIRecordsLostError` | 那次迭代已经执行，只是读数丢失。整个 phase 重新测量，总共最多测 3 次，每次把缓冲扩大 4 倍 |
| 没有丢弃，但有 kernel 无法对应到迭代号 | `_OffThreadLaunchError` | 这个 kernel 由一个没有标记迭代号的线程发起 |
| 没有丢弃，但某次迭代没有任何 kernel | `_CUPTIAttributionError` | 这次调用没有在设备上执行 |

## 被测量的量 {#what-is-measured}

**`device_busy_ms` 是一次调用产生的全部 kernel 在设备上执行区间的并集长度。** CUPTI 的 kernel 记录给出设备上的执行起止时间，不含 CPU 发起这次调用的开销。分三种情形：

- **单 kernel 的调用**：这个 kernel 在设备上的执行时长。
- **多 kernel 的调用**：各执行区间的并集，即设备上至少有一个属于该调用的 kernel 在执行的总时长。两个并发的 kernel 不计为两份，因为两份之和是 SM 时间，不是设备忙碌的时间。
- **kernel 之间的空隙**：不计入。

空隙不计入，是因为无法确定它的成因。设备在那段时间确实空闲，但成因可能是 op 自身的数据依赖，也可能是 CPU 尚未发出下一个 kernel，两者在 CUPTI 的记录中没有区别。成因无法区分的量不能用来判断一个实现的好坏。

`tflops` 与 `bandwidth_tbs` 的分母也是这个量。它们描述设备执行期间达到的吞吐，分母若包含调用内的空闲时间，会把吞吐系统性地压低。

这样定义的量不受主机快慢的影响。把 CUPTI 的采集缓冲从 256 KB 换成 32 MB 后，同一个三 kernel 调用的 `latency_ms` 中位数从 35 us 增加到 2068 us，`device_busy_ms` 始终是 19.1 us。主机晚发出 kernel 不改变任何 kernel 的执行时长，只是把它们在时间轴上推远，并集长度不变。

## 为什么不是墙钟时间 {#why-not-wall-clock}

在 decode 尺度上，op 的执行时间可能短于发起它的那次 Python 调用。对同一个 3 us 的 kernel，四种测法得到四个读数：

| 测法 | 读数 |
| --- | --- |
| CUPTI 的 kernel 记录 | 1.95 us |
| 逐次一对 CUDA events | 6.03 us |
| 整个循环一对 events，再除以迭代数 | 6.07 us |
| CUDA graph 重放 | 4.30 us |

设备上的实际执行时间是 1.95 us，event 方案读出的 6 us 是 CPU 发起下一次调用的间隔。**这是 TileOPs 用 CUPTI 计时的唯一理由**，也是退回 CUDA events 的那一行不能与其余行比较的原因：这一行的 `device_busy_ms` 与 `latency_ms` 记录同一个数，`timing` 字段记为 `cuda-events`。

## 比较多个实现 {#comparing}

在同一个用例中比较多个实现时，`bench.Runner.compare()` 按 A B C C B A 的顺序让每个实现各运行两段，两段样本合并后取中位数。

在固定顺序下，先运行和后运行的实现处在不同的时钟与温度状态，这个差别会被误读为实现之间的差别。对称顺序让每个实现的两段分别位于全程的前半和后半，单调漂移在一阶上相互抵消。有两个细节：

- **预算拆分，不翻倍。** 每段预热 12.5 ms、测量 50 ms，迭代次数的上下限各取一半。对称顺序的目的是抵消漂移，不是增加样本，因此样本量与单个实现计时时相当。
- **两段的计时方法必须一致。** 一段使用 CUPTI、另一段退回 CUDA events 时直接报错，不合并结果，否则一个中位数会混合两种测量方法。

## 什么时候要改测法 {#when-to-change}

默认情形下不需要任何处理：一次调用只发出一个 kernel，经由 Op 接口，使用 `bench_kernel`，并且进程中没有其他线程使用 GPU。当前多数 op 属于这种情形。以下八种情形需要单独处理：

| 情形 | 不处理的后果 | 处理方法 |
| --- | --- | --- |
| 被测闭包中包含 `Tensor.backward` 或 `torch.autograd.grad` | 反向的 kernel 由 autograd 引擎的线程发出，无法对应到迭代号，整个用例报错，不产出数字 | 单个融合节点用 `backward_of(out)` 直接驱动；多节点链改用 `torch.autograd.set_multithreading_enabled(False)` |
| 进程中有其他线程在使用 GPU，或被测闭包自身使用了 CUPTI 的 `CUSTOM0` external id | 那些 kernel 无法对应到迭代号，或迭代号被闭包覆盖，同样报错 | 由被计时的调用自己启动它的工作；external id 改用 `CUSTOM1` / `CUSTOM2` |
| op 依靠 `copy_` 回写才产出结果，例如原地 elementwise 与 MoE 的写回 | 计时会采集这次拷贝，但默认不计入 `device_busy_ms`，而是另记在 `uncounted_copy_ms` 中，读数因此偏小 | 在该 op 位于 `benchmarks/_cases/` 的 case 注册项中设置 `count_copies=True`，这个用例中所有实现的读数都会计入拷贝 |
| 某个实现原地写入自己的输入，或在调用之间保留状态 | 之后的迭代从不同的数据开始；共享的 `case.inputs` 被改写时，`compare()` 在计时前报错 | 为被改写的参数准备私有副本，并在 `reset` 中恢复，写法见 [`bench.Implementation`](user-guide/benchmark/writing.md#implementation)；`reset` 在清空 L2 之前执行，不计入读数 |
| 一次调用发出多个 kernel | kernel 之间的空隙计入 `latency_ms`，用它与融合实现比较时，空隙算在多 kernel 的一方 | 结论只依据 `device_busy_ms`；`latency_ms` 只在两行的 `n_kernels` 相同时可比 |
| 单次调用超过 10 ms | 迭代次数停在下限 10，墙钟时间远超 100 ms 的预算，10 个样本给出的 p10/p90 很粗 | 接受更长的墙钟时间，或显式指定迭代次数并写明样本量 |
| 需要新增一个 kernel 级的 benchmark | 这个 op 没有 spec，形状与 roofline 只能手写，spec 校验器也检查不到它 | 经由 Op 接口测量，并补一份 [spec](user-guide/manifest/index.md) |
| 新增一个外部基线 | 如果基线的输入转换被移出它的计时区间，相当于本仓库替它承担了这部分时间 | 转换保留在基线的计时区间内。这个基线是该 benchmark 存在的理由时，要求依赖必须存在，缺失时让 import 失败 |
