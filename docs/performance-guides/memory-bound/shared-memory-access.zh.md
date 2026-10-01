# 优化 shared memory 访问

数据经过 shared memory 中转时，shared memory 上的 access pattern 也需要考虑：数据从 global memory 写入 shared memory，再从 shared memory 读入寄存器，这两步都访问 shared memory。本页说明这两步上的 bank conflict：它由什么决定，如何用 pad 消除，以及 pad 的取值如何计算。

本页的实测条件是：H200，SM 时钟锁在 1830 MHz，输入大于 L2 的 60 MiB，block 数足以填满整卡。在这组条件之外，结论可能反转，判据见[优化 global memory 访问](global-memory-access.md#regime)。

## shared memory 的 bank 结构 {#bank-conflict}

shared memory 由 32 个 bank 构成，每个 bank 宽 4 字节。将 shared memory 的地址空间按 4 字节划分成 **word**（下文一律按 word 计数），一个地址落在哪个 bank 上，由 `(字节地址 / 4) mod 32` 决定。一个 bank 每周期只能处理一个 word 的访问。多个线程并发访问 shared memory 时，只要访问落在不同的 bank 上，就在同一个周期内一起完成。多个线程落在同一条 bank 上时，分三种情形：

1. **访问的是不同的 word** —— 硬件把这次请求拆成若干次无冲突的请求依次完成，拆分的次数就是**冲突的路数**。
2. **读取的是同一个 word** —— 任意两个线程只要落在同一个 word 内（即使取的是其中不同的字节），这个 word 会被广播给所有请求它的线程，不产生冲突。分布在不同 bank 上的多个广播还会合并为一次 multicast。
3. **写入的是同一个地址** —— 只有一个写入生效，是哪一个未定义。

访问 shared memory 的 access pattern 应当尽可能做到无 bank conflict。

## 冲突路数由什么决定

以下面的一般情形为例：一维数组 `sh` 位于 shared memory，每个线程读取其中连续的 `chunk` 个元素。

```python
sh = T.alloc_shared((threads * chunk,), dtype)   # 一维数组，threads * chunk 个元素

for c in T.serial(chunk):
    acc[0] = acc[0] * sh[tx * chunk + c]         # 线程 tx 读自己那一段
```

一个 warp 的 32 个线程同步执行这个循环，同一次迭代中 `c` 对它们取同一个值，`tx` 取 0 到 31，因此 32 个线程以相等的间隔访问 shared memory。例如 `chunk = 64` 时，这 32 个线程在同一次迭代里读的是第 `c`、第 `64 + c`、第 `128 + c`、…… 第 `1984 + c` 个元素。相邻两个线程相差 `chunk` 个元素，这个差值称为这段访问的 **stride**，记作 $S$，换算成 word 是 $S = \text{chunk} \times E / 4$ 个（$E$ 是元素的字节数）。stride 与数组的维数、下标的写法都无关。

另一个变量是访存指令的位宽。以向量化的方式一次访问 $w$ 个连续的 word，$w$ 取 1、2、4，对应 32 bit、`float2` 的 8 字节、`float4` 的 16 字节。于是线程 $t$ 读的是第 $St$ 到 $St + w - 1$ 个 word。

以下两条前提成立时，冲突路数可以由上述 bank 结构直接算出：

1. **$S$ 是整数个 word。** fp16 的 `chunk` 取奇数时，$S$ 含半个 word，无法取公约数，只能按字节地址逐个算出每个线程落在哪条 bank 上，再计数。
2. **32 个线程请求的 $32w$ 个 word 互不相同**，即 $S \ge w$。线程落在同一个 word 上时走广播，不占额外的周期，此时按 $32w$ 个 word 计数不成立；$S < w$ 时相邻线程的向量区间彼此重叠，同样不成立。

整个 warp 要取 $32w$ 个 word，而 shared memory 每周期最多处理 32 个，所以这条指令至少需要 $w$ 个周期。这个下界由位宽决定，与地址无关；冲突指超出这个下界的部分。$St \bmod 32$ 只取 $g = \gcd(S, 32)$ 的倍数，即只落在 $32/g$ 条 bank 上，每条被访问 $g$ 次；再叠加 $j = 0, \dots, w-1$ 的平移。$g$ 与 $w$ 都是 2 的幂，必有一个整除另一个，因此分两种情形：

| | bank 的落点 | 周期数 | 冲突 |
| --- | --- | --- | --- |
| $g \le w$ | 32 条 bank 各被请求 $w$ 次 | $w$，正好是下界 | 无 |
| $g > w$ | 只有 $(32/g) \cdot w$ 条被访问，各 $g$ 次 | $g$ | $g / w$ 路 |

$$\text{冲突路数} \ \ge\ \max\left(1,\ \frac{\gcd(S,\ 32)}{w}\right)$$

标量读（$w = 1$）时它就是 $\gcd(S, 32)$：$S$ 与 32 互素时 32 个线程铺满 32 条 bank，无冲突；$S$ 是 32 的倍数时全部落在同一条 bank 上，32 路串行。fp16、`chunk = 64` 属于后者，$S$ 是 128 字节即 32 个 word。

上式写成不等式，原因是最后一步假定硬件能把任意一组无冲突的 word 放进一个周期，而 NVIDIA 没有公开 64 bit 与 128 bit 访问时 lane 的实际分组方式。

## 用 pad 改变 stride

stride 由 `chunk` 决定，而 `chunk` 通常由算法决定，往往不能任意改动。发生 $N$ 路冲突时，一种做法是在每段末尾增加 `pad` 个元素，使 stride 变为 `chunk + pad`，`gcd` 随之改变，冲突路数也随之改变。例如 fp16、`chunk = 64` 时，只需加 2 个元素，stride 就从 32 个 word 变为 33 个，与 32 互素，冲突完全消失。

<figure class="bank-conflict" markdown="1">

<svg class="tf-bank" viewBox="0 0 520 340" role="img" aria-label="fp16、每段 64 个元素时，一个 warp 的 32 个线程落在 32 条 shared memory bank 上的分布。不加 pad 时全部落在 bank 0，是 32 路冲突；pad = 2 时铺满 32 条 bank，无冲突；pad = 4 时两个线程共用一条 bank，是 2 路；pad = 8 时四个共用一条，是 4 路。">
<text class="bk-title" x="0" y="43.0">pad = 0</text>
<text class="bk-sub" x="0" y="58.0">stride = 32 word</text>
<text class="bk-sub" x="0" y="72.0">gcd(32, 32) = 32</text>
<rect class="bk-cell bk-cell--many" x="150.0" y="30.0" width="11.0" height="20.0"/>
<text class="bk-count" x="155.5" y="43.5" text-anchor="middle">32</text>
<rect class="bk-cell" x="161.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="172.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="183.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="194.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="205.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="216.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="227.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="238.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="249.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="260.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="271.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="282.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="293.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="304.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="315.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="326.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="337.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="348.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="359.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="370.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="381.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="392.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="403.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="414.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="425.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="436.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="447.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="458.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="469.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="480.0" y="30.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="491.0" y="30.0" width="11.0" height="20.0"/>
<text class="bk-note" x="150.0" y="66.0">用到 1 / 32 条 bank，最多的一条要服务 32 个线程 —— 32 路冲突</text>
<text class="bk-title" x="0" y="117.0">pad = 2</text>
<text class="bk-sub" x="0" y="132.0">stride = 33 word</text>
<text class="bk-sub" x="0" y="146.0">gcd(33, 32) = 1</text>
<rect class="bk-cell bk-cell--one" x="150.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="161.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="172.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="183.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="194.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="205.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="216.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="227.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="238.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="249.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="260.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="271.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="282.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="293.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="304.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="315.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="326.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="337.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="348.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="359.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="370.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="381.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="392.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="403.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="414.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="425.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="436.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="447.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="458.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="469.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="480.0" y="104.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--one" x="491.0" y="104.0" width="11.0" height="20.0"/>
<text class="bk-note" x="150.0" y="140.0">用到 32 / 32 条 bank，最多的一条要服务 1 个线程 —— 无冲突</text>
<text class="bk-title" x="0" y="191.0">pad = 4</text>
<text class="bk-sub" x="0" y="206.0">stride = 34 word</text>
<text class="bk-sub" x="0" y="220.0">gcd(34, 32) = 2</text>
<rect class="bk-cell bk-cell--many" x="150.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="155.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="161.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="172.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="177.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="183.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="194.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="199.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="205.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="216.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="221.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="227.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="238.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="243.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="249.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="260.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="265.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="271.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="282.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="287.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="293.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="304.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="309.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="315.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="326.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="331.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="337.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="348.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="353.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="359.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="370.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="375.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="381.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="392.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="397.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="403.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="414.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="419.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="425.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="436.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="441.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="447.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="458.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="463.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="469.0" y="178.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="480.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-count" x="485.5" y="191.5" text-anchor="middle">2</text>
<rect class="bk-cell" x="491.0" y="178.0" width="11.0" height="20.0"/>
<text class="bk-note" x="150.0" y="214.0">用到 16 / 32 条 bank，最多的一条要服务 2 个线程 —— 2 路冲突</text>
<text class="bk-title" x="0" y="265.0">pad = 8</text>
<text class="bk-sub" x="0" y="280.0">stride = 36 word</text>
<text class="bk-sub" x="0" y="294.0">gcd(36, 32) = 4</text>
<rect class="bk-cell bk-cell--many" x="150.0" y="252.0" width="11.0" height="20.0"/>
<text class="bk-count" x="155.5" y="265.5" text-anchor="middle">4</text>
<rect class="bk-cell" x="161.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="172.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="183.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="194.0" y="252.0" width="11.0" height="20.0"/>
<text class="bk-count" x="199.5" y="265.5" text-anchor="middle">4</text>
<rect class="bk-cell" x="205.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="216.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="227.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="238.0" y="252.0" width="11.0" height="20.0"/>
<text class="bk-count" x="243.5" y="265.5" text-anchor="middle">4</text>
<rect class="bk-cell" x="249.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="260.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="271.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="282.0" y="252.0" width="11.0" height="20.0"/>
<text class="bk-count" x="287.5" y="265.5" text-anchor="middle">4</text>
<rect class="bk-cell" x="293.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="304.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="315.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="326.0" y="252.0" width="11.0" height="20.0"/>
<text class="bk-count" x="331.5" y="265.5" text-anchor="middle">4</text>
<rect class="bk-cell" x="337.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="348.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="359.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="370.0" y="252.0" width="11.0" height="20.0"/>
<text class="bk-count" x="375.5" y="265.5" text-anchor="middle">4</text>
<rect class="bk-cell" x="381.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="392.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="403.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="414.0" y="252.0" width="11.0" height="20.0"/>
<text class="bk-count" x="419.5" y="265.5" text-anchor="middle">4</text>
<rect class="bk-cell" x="425.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="436.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="447.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell bk-cell--many" x="458.0" y="252.0" width="11.0" height="20.0"/>
<text class="bk-count" x="463.5" y="265.5" text-anchor="middle">4</text>
<rect class="bk-cell" x="469.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="480.0" y="252.0" width="11.0" height="20.0"/>
<rect class="bk-cell" x="491.0" y="252.0" width="11.0" height="20.0"/>
<text class="bk-note" x="150.0" y="288.0">用到 8 / 32 条 bank，最多的一条要服务 4 个线程 —— 4 路冲突</text>
<text class="bk-axis" x="155.5" y="22.0" text-anchor="middle">bank 0</text>
<text class="bk-axis" x="496.5" y="22.0" text-anchor="end">31</text>
<text class="bk-scale" x="0" y="337.0">fp16、每段 64 个元素、一个 warp 的 32 个线程。格内数字是落在该 bank 上的线程数，空格表示没有线程落上去。</text>
</svg>

<figcaption>四张图是同一个 kernel 在四个 pad 取值下，一个 warp 的 32 个线程在 32 条 bank 上的落点。格内数字是落在这条 bank 上的线程数，最大的那个数就是冲突路数。<code>pad = 0</code> 时 32 个线程全部落在 bank 0；加 2 个元素之后 stride 变成 33 个 word，与 32 互素，32 个线程正好铺满 32 条 bank。</figcaption>

</figure>

## pad 该取多少

判据由上面的式子与段起点的对齐要求共同给出，**两条都要满足**：

| 访问位宽 | $w$ | 冲突路数要求 | 段起点的对齐要求 | fp32、`chunk = 64` 上最快的 pad |
| --- | --- | --- | --- | --- |
| 32 bit，标量读 | 1 | $\gcd(S, 32) = 1$ | 4 字节 | `pad = 1`（$S = 65$ word） |
| 64 bit，`float2` | 2 | $\gcd(S, 32) \le 2$ | 8 字节 | `pad = 2`（$S = 66$ word） |
| 128 bit，`float4` | 4 | $\gcd(S, 32) \le 4$ | 16 字节 | `pad = 4`（$S = 68$ word） |

**位宽越宽，对冲突路数的要求越松，对对齐的要求越紧**。两条要求方向相反，因此不能只看其中一条。奇数 pad 对标量读最优，对向量化读最差，因为它使段起点错开 4 字节。128 bit 的 shared 读要求 16 字节自然对齐，对齐不足时这条指令的行为是未定义的；编译期能看出对齐不足时，编译器会退回窄指令，下面实测里 $\gcd = 1$ 那一行就是这种情形。

下表是实测结果（H200，fp32，`chunk = 64`，消费循环重复 32 遍使 shared 一侧成为瓶颈，耗时随遍数成比例），每个数值是该组合的耗时相对同一位宽最快一行的倍数：

| $\gcd(S, 32)$ | 32 bit | 64 bit | 128 bit | 下界预测（32 / 64 / 128） |
| --- | --- | --- | --- | --- |
| 1 | **1.00** | 1.10 | 2.27 | 1 / 1 / 1 |
| 2 | 1.72 | **1.00** | 1.01 | 2 / 1 / 1 |
| 4 | 3.29 | 1.81 | **1.00** | 4 / 2 / 1 |
| 8 | 6.52 | 3.55 | 4.04 | 8 / 4 / 2 |
| 16 | 12.97 | 7.62 | 7.99 | 16 / 8 / 4 |
| 32 | 25.82 | 13.97 | 15.71 | 32 / 16 / 8 |

对角线上的三个 1.00 对应上一张表推荐的三个 pad。$g \le w$ 一侧的下界是紧的。$g > w$ 一侧的下界不紧，128 bit 的实测恰好是下界的 2 倍（4.04 对 2、7.99 对 4、15.71 对 8）；成因需要更底层的证据才能确定，NVIDIA 未公开这两种位宽下 lane 的分组方式。$\gcd = 1$ 那一行的 2.27 是对齐造成的：$S = 65$ word 即 260 字节，不是 16 的倍数。

## pad 要按 chunk 算 {#pad-per-chunk}

在某些 `chunk` 上，固定字节数的 pad 会重新出现最坏情形的冲突。段起点的对齐要求把候选限制在 16 字节的整数倍，记 pad 为 $k$ 个 16 字节（$k = 1, 2, \dots$），于是

$$S = \frac{\text{chunk} \times E}{4} + 4k \ \text{word}$$

$k$ 固定时，$\gcd(S, 32)$ 随 `chunk` 变化。下表列出 fp16 与 bf16（$E = 2$）的几个 `chunk`：

| chunk | $S$（$k = 1$） | $\gcd(S, 32)$ | $S$（$k = 2$） | $\gcd(S, 32)$ |
| --- | --- | --- | --- | --- |
| 32 | 20 | **4** | 24 | 8 |
| 56 | 32 | 32 | 36 | **4** |
| 64 | 36 | **4** | 40 | 8 |
| 72 | 40 | 8 | 44 | **4** |
| 128 | 68 | **4** | 72 | 8 |

`chunk = 56`、$k = 1$ 时 $S$ 恰好是 32 个 word，与 `pad = 0` 一样全部落在同一条 bank 上，加了 pad 也没有消除冲突。做法是在 $k = 1$ 与 $k = 2$ 两个候选里取 $\gcd(S, 32)$ 小的那个。$\text{chunk} \times E$ 是 16 的倍数时，两者必有一个把 $\gcd$ 降到 4：记 $q = \text{chunk} \times E / 16$，则 $S = 4(q + k)$，$\gcd(S, 32) = 4 \gcd(q + k, 8)$，而 $q + 1$ 与 $q + 2$ 一奇一偶。

实测（H200，SM 时钟锁在 1830 MHz，bf16，CUPTI 设备耗时，每次迭代前清 L2，取 200 次的中位数，镜像 `ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-afcebed1-dev`）。四行都选取线程数使 `chunk` 等于 56，同一行的两列只差 pad：

| 输入 | 线程数 × chunk | $k = 1$（pad 8 个元素）<br>TB/s | $k = 2$（pad 16 个元素）<br>TB/s |
| --- | --- | --- | --- |
| $2048 \times 3584$ | 64 × 56 | 1.38 | **3.07**{ .win } |
| $2048 \times 7168$ | 128 × 56 | 1.58 | **3.29**{ .win } |
| $1024 \times 14336$ | 256 × 56 | 1.51 | **3.05**{ .win } |
| $512 \times 28672$ | 512 × 56 | 1.32 | **2.33**{ .win } |

这四个宽度分别是 Qwen2-7B 的 hidden size、Llama-3-70B 的 hidden size、Llama-3-8B 与 Llama-3-70B 的 FFN 中间维，都取自实际模型。

这组测量中 shared 一侧是 64 bit 读：`cuobjdump` 显示 $S = 36$ word 时每线程 16 条 `LDS.64`，即 $w = 2$，所以路数是 $\gcd(S, 32) / 2$，两列分别是 16 路与 2 路。路数相差 8 倍而带宽只差 2.2 倍，原因是 2 路那一列的瓶颈已经回到 DRAM。段起点错开不足 8 字节时，编译器改用 `LDS`（$w = 1$）。位宽由编译器决定，不由声明 pad 的开发者决定，所以上表两列都要实测，不能只计算 $\gcd$。

## 实测扫描

下面的实测中，每个线程逐个元素读取自己那一段（$w = 1$），`chunk` 取 2 的幂，因此「冲突路数由什么决定」一节的两条前提都成立。

H200，SM 时钟锁在 1830 MHz。fp16，输入 $65536 \times 4096$（512 MB，大于 60 MiB 的 L2）。kernel 把整行搬进 shared memory，每个线程一段 `chunk + pad` 个元素，逐段做串行前缀积再写回，读加写共 1 GB。括号内是上面公式预测的冲突路数：

| chunk | 线程数 | pad = 0<br>TB/s | pad = 2<br>TB/s | pad = 4<br>TB/s | pad = 8<br>TB/s | pad = 16<br>TB/s |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 256 | 1.63（8 路） | 2.99（1 路） | **3.29**{ .win }（2 路） | 2.58（4 路） | 0.86（16 路） |
| 32 | 128 | 0.89（16 路） | 3.32（1 路） | **3.35**{ .win }（2 路） | 2.57（4 路） | 1.54（8 路） |
| 64 | 64 | 0.46（32 路） | 3.63（1 路） | **3.70**{ .win }（2 路） | 2.74（4 路） | 1.62（8 路） |
| 128 | 32 | 0.46（32 路） | 3.25（1 路） | **3.31**{ .win }（2 路） | 2.76（4 路） | 1.61（8 路） |

20 个配置里，1 路与 2 路在 3.0 以上，4 路降到 2.6 附近，8 路及以上跌到 1.6 以下。带宽随预测的路数单调下降，所以在这组条件下公式可以直接用来缩小 pad 的候选。

## 使用时的注意事项

1. **公式给出候选，最终取值由实测决定。** 上表里 1 路与 2 路都在 3.0 以上，4 路降到 2.6 附近，8 路及以上跌到 1.6 以下，所以公式的用处是把候选缩到「算出来不超过 $w$ 路」的那几个。这几个候选之间需要实测：四组 `chunk` 上 2 路都略高于 1 路，但差距从 0.9% 到 10% 不等，不构成一条可以照搬的规则。

2. **修改 pad 之后需要重新扫描 `chunk`。** `pad = 0` 那一列最优的是 `chunk = 16`（1.63），`pad = 4` 那一列最优的是 `chunk = 64`（3.70）。前一列的排序主要由冲突路数决定：`chunk = 16` 的冲突是 8 路，`chunk = 64` 的冲突是 32 路。消除冲突之后四组都是 2 路，最优点随之改变。表中线程数与 `chunk` 联动（两者之积恒为行宽 4096），所以最优点改变的原因不只是 `chunk`，占用率与循环长度也在变化；能确定的只是修改 pad 之后 `chunk` 的排序会改变。

3. **`chunk` 改变后需要重新计算 pad。** 把 pad 写成固定字节数，等于假定 $\gcd(S, 32)$ 与 `chunk` 无关，而[pad 要按 chunk 算](#pad-per-chunk)一节的表说明两者相关：`chunk = 56`、$k = 1$ 时 $S$ 回到 32 个 word。把 pad 写成 `chunk` 的函数，即在 $k = 1, 2$ 中取 $\gcd(S, 32)$ 较小的那个，才与本页的判据一致。

4. **本页只适用于数据经过 shared memory 的情形。** 按照[优化 global memory 访问](global-memory-access.md#coalescing)中的取舍，$V$ 小时使用向量化的 blocked，数据直接进入寄存器，不经过 shared memory，因此不存在 bank 冲突。本页适用于两种情形：$V$ 大到寄存器压力压低占用率、改用 staged 时；整行需要由 block 内所有线程共享时。

下面两段是 shared 缓冲的声明，两者只有 stride 不同。反例的 stride 恰好是 32 个 word 的倍数：

```python
sh = T.alloc_shared((threads * chunk,), dtype)          # stride = chunk 个元素
```

正例按 `chunk` 计算 pad，在 16 字节的整数倍中取 $\gcd(S, 32)$ 最小的那个：

```python
import math

def pick_pad(chunk: int, elem_bytes: int) -> int:
    """16 字节的整数倍里，gcd(S, 32) 最小的 pad，单位是元素。"""
    return min(
        (16 // elem_bytes, 32 // elem_bytes),                    # k = 1、k = 2
        key=lambda pad: math.gcd((chunk + pad) * elem_bytes // 4, 32),
    )

pad = pick_pad(chunk, elem_bytes)                                # 候选之间仍要实测
sh = T.alloc_shared((threads * (chunk + pad),), dtype)           # stride = chunk + pad
```
