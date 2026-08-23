# benchmark 的计时方法

nightly benchmark 给每个算子的每个 workload 各测一行，报出的 `device_busy_ms` 是一次调用产生的全部 kernel 在设备上执行区间的并集。CUPTI 记下每个 kernel 在设备上的执行起止，按 external correlation id 归到某一次迭代；每次迭代之前清空 L2，25 ms 预热、100 ms 测量，取中位数。

**所以表里的每个数都是设备执行 kernel 的时间：不含 CPU 发起调用的开销，也不含 kernel 之间的空隙。读表到这里就够了。**{ .keystone }

余下各节是补充，按需要查：

- [一次测量的流程](#how-it-runs) —— 伪代码，加上校准、迭代次数、清 L2、归属、失败即停五件事。
- [被测量的量](#what-is-measured) —— `device_busy_ms` 的定义，以及 kernel 之间的空隙为什么不计入。
- [为什么不是墙钟时间](#why-not-wall-clock) —— decode 尺度上 CUDA events 测不出小 kernel。
- [什么时候要改测法](#when-to-change) —— 自己写 benchmark 时才需要，连同这套测法测不到的几种情形。

下文的数字都在 H200 上实测，镜像为 `tileops-runner:cu132-torch2.13`。

## 一次测量的流程 {#how-it-runs}

```python
from benchmarks.timing import bench_kernel

samples = bench_kernel(op, args=(x, weight))   # 每次迭代一个 Sample
```

写 benchmark 时一般不直接调它，而是走 `ManifestBenchmark.profile()` 或 `.compare()`，那一层在这些样本之上再取中位数、算派生列。

`bench_kernel` 内部分三段 —— 采集、归属、计量：

```python
# 采集：每次调用在自己的迭代号下执行
with _phase_session():                          # kernel + 映射 + launch 类 API
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

这段代码里的五处安排，各有其原因：

1. **校准。** 先跑 3 次，估出单次调用的耗时。
2. **换算迭代次数。** 预热 25 ms、测量 100 ms 的时间预算除以单次耗时，结果钳在 `[10, 200]`。短算子采样次数多，长算子不必跑满 200 次。
3. **每次迭代之前清 L2，并等设备排空。** 不清 L2，第一次迭代从 HBM 读、之后每次都从 L2 读，中位数报出来的是缓存全命中的最好情况；等排空是为了让上一次迭代不与这一次重叠。
4. **采集与归属。** 每次迭代把自己的迭代号标记为 CUPTI 的 external correlation id，这段区间内发出的每次 launch 都带上它；kernel 记录携带的 correlation id 再经这层映射回到迭代号。**归属不看时间戳** —— 一个 kernel 属于哪次迭代写在它的记录里，与它何时执行无关，因此比主机开销更短的 kernel 与长 kernel 归属得一样可靠，一次调用的 kernel 数在迭代之间变化也照样测得出来。
5. **失败即停。** 三种归属失败各报一种错，不产出数字：

| 情形 | 报什么 | 含义 |
| --- | --- | --- |
| CUPTI 丢了记录 | `_CUPTIRecordsLostError` | 读数丢了，那次迭代其实跑过 —— 整个 phase 重测，最多 3 次，每次把缓冲要大 4 倍 |
| 没有丢弃，但有 kernel 带不上迭代号 | `_OffThreadLaunchError` | 某个没有标记迭代号的线程发起了它 |
| 没有丢弃，某次迭代一个 kernel 都没有 | `_CUPTIAttributionError` | 这次调用没上设备 |

## 被测量的量 {#what-is-measured}

**`device_busy_ms`：一次调用产生的全部 kernel，在设备上执行区间的并集长度。** CUPTI 的 kernel 记录给出的 start 与 end 是设备上的执行起止，不含 CPU 发起这次调用的开销。三种情形：

- **单 kernel 的调用** —— 就是这个 kernel 在设备上的执行时长。
- **多 kernel 的调用** —— 各执行区间的并集，也就是设备上至少有一个该调用的 kernel 在执行的总时长。并发执行的两个 kernel 不因并发计成两份，那是 SM 时间，不是设备忙的时间。
- **kernel 之间的空隙** —— 不计入。

空隙不计入，是因为它归不了因。那段时间设备确实空闲，但成因既可能是算子自身的数据依赖，也可能是 CPU 还没发出下一个 kernel，两者在 CUPTI 的记录里没有区别。分不清成因的量，不能用来判断一个实现的好坏。

`tflops` 与 `bandwidth_tbs` 的分母也是这个量：它们描述设备执行期间达到的吞吐，分母含调用内的空闲会把它系统性压低。

这样定义的量对主机的快慢免疫。把 CUPTI 的采集缓冲从 256 KB 换成 32 MB，同一个三 kernel 调用的 `latency_ms` 中位数从 35 us 涨到 2068 us，`device_busy_ms` 始终是 19.1 us —— 主机晚发不改变任何 kernel 自身的执行时长，只是把它们在时间轴上推远，各区间的并集不变。

## 为什么不是墙钟时间 {#why-not-wall-clock}

decode 尺度上，一个算子的执行时间可能短于发起它的那次 Python 调用。同一个 3 us 的 kernel，四种测法读出四个数：

| 测法 | 读数 |
| --- | --- |
| CUPTI 的 kernel 记录 | 1.95 us |
| 逐次一对 CUDA events | 6.03 us |
| 整个循环一对 events，再除以迭代数 | 6.07 us |
| CUDA graph 重放 | 4.30 us |

设备上实际执行了 1.95 us，event 方案读出的 6 us 是 CPU 发起下一次调用的节奏，不是这个 kernel 的执行时间。**这是 TileOPs 用 CUPTI 计时的唯一理由**，也是为什么退回 CUDA events 之后那一行不能与其余行比较：`device_busy_ms` 与 `latency_ms` 记同一个数，`timing` 字段记为 `cuda-events`。

## 比较多个实现

同一个用例里比较几个实现时，`compare()` 按 A B C C B A 的顺序各跑两段，每个实现的两段样本合并后取中位数。

固定顺序下，先跑和后跑的实现处在不同的时钟与温度状态，这个差别会被读成实现之间的差别；对称顺序让每个实现的两段分列全程的前后两半，单调漂移一阶抵消。两个细节：

- **预算是拆开的，不是翻倍。** 每段 12.5 ms 预热、50 ms 测量，迭代上下限也各取一半 —— 要的是对称，不是更多样本，所以样本量与单实现计时相当。
- **两段的计时方法必须一致。** 一段走 CUPTI、另一段退回 CUDA events 时直接报错，不合并，否则一个中位数横跨两种测量。

## 什么时候要改测法 {#when-to-change}

默认情形什么都不用管：一次调用只发一个 kernel、经由 Op 接口、走 `bench_kernel`、进程里没有别的线程在用 GPU。当前多数算子都是这个情形。下面七种要停下来处理：

| 你的情形 | 不处理会怎样 | 你要做的 |
| --- | --- | --- |
| 被测闭包里包了 `Tensor.backward` 或 `torch.autograd.grad` | 反向的 kernel 由 autograd 引擎的线程发出，带不上迭代号，整个用例报错，不产出数字 | 单个融合节点用 `backward_of(out)` 直接驱动；多节点链改用 `torch.autograd.set_multithreading_enabled(False)` |
| 进程里有别的线程在用 GPU，或被测闭包自己用了 CUPTI 的 `CUSTOM0` external id | 那些 kernel 带不上迭代号，或者迭代号被闭包盖掉，同样报错 | 让被计时的调用自己启动它的工作；external id 改用 `CUSTOM1` / `CUSTOM2` |
| 算子靠 `copy_` 回写才产出结果，例如原地 elementwise 与 MoE 的写回 | 那次拷贝走 `cudaMemcpyAsync`，只采集 kernel 活动的计时看不见它，读数偏小，而且不报错 | 采集 `MEMCPY` / `MEMSET` 当检查用，确认窗口内没有 |
| 一次调用发多个 kernel | kernel 之间的空隙落在 `latency_ms` 里，用它与融合实现比较，空隙记在你这一方 | 结论只用 `device_busy_ms`；`latency_ms` 仅在两行 `n_kernels` 相同时可比 |
| 单次调用超过 10 ms | 迭代次数被下限 10 顶住，墙钟远超 100 ms 的预算，10 个样本给出的 p10/p90 很粗 | 接受更长的墙钟，或显式指定迭代次数并写明样本量 |
| 想加一个 kernel 级的 benchmark | 这个算子没有 spec，形状与 roofline 只能手写，spec 校验器也查不到它 | 经 Op 接口测，并补一份 [spec](manifest.md) |
| 加一个外部基线 | 基线的输入转换若移出它的计时区间，等于本仓库替它承担了这部分时间 | 转换留在基线的计时区间内。这个基线是 benchmark 存在的理由时，要求依赖存在、让 import 失败 |
