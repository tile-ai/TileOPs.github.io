# 优化 shared memory 访问

数据经过 shared memory 中转时，多了一处需要考虑 access pattern 的地方：从 global memory 写进来、再从 shared memory 读进寄存器，两步都在访问 shared memory。这一页讲这两步上的 bank conflict —— 它由什么决定、怎么用 pad 消掉、以及 pad 该按什么算。

这一页的实测都在 H200 上、SM 时钟锁在 1830 MHz、输入大于 L2 的 60 MiB、block 数足以填满整卡 —— 这组条件之外结论可能反转，判据见[优化 global memory 访问](global-memory-access.md#regime)。

## shared memory 的 bank 结构 {#bank-conflict}

shared memory 由 32 个 bank 构成，每 bank 宽 4 个字节。将 shared memory 的地址空间按 4 字节划分成 **word**（下文一律按 word 计数），一个地址落在哪个 bank 上，由 `(字节地址 / 4) mod 32` 决定。一个 bank 每周期只能处理一个 word 的访问，多个线程并发访问 shared memory 时，只要访问落在不同的 bank 上，它们就在同一个周期一起完成。多个线程落在同一条 bank 上时，又分三种情形：

1. **访问的是不同的 word** —— 硬件把这次请求拆成若干次无冲突的请求依次完成，拆分的次数就是**冲突的路数**。
2. **读取的是同一个 word** —— 任意两个线程只要落在同一个 word 内（哪怕取的是其中不同的字节），这个 word 会被广播给所有请求它的线程，不产生冲突。分布在不同 bank 上的多个广播还会合并为一次 multicast。
3. **写入的是同一个地址** —— 只有一个写入生效，是哪一个未定义。

访问 shared memory 的 access pattern 应当尽可能做到无 bank conflict。

## 冲突路数由什么决定

我们考虑一种不失通用性的情况。下面这个一维数组 `sh` 在 shared memory 上，每个线程读取其中连续的 `chunk` 个元素：

```python
sh = T.alloc_shared((threads * chunk,), dtype)   # 一维数组，threads * chunk 个元素

for c in T.serial(chunk):
    acc[0] = acc[0] * sh[tx * chunk + c]         # 线程 tx 读自己那一段
```

一个 warp 的 32 个线程同步执行这个循环，同一次迭代里 `c` 对它们取同一个值、`tx` 取 0 到 31：这时 32 个线程以等间隔访问 shared memory。例如 `chunk = 64` 时，这 32 个线程在同一次迭代里读的是第 `c`、第 `64 + c`、第 `128 + c`、…… 第 `1984 + c` 个元素。相邻两个线程相差 `chunk` 个元素，这个差值称为这段访问的 **stride**，记作 $S$，换算成 word 是 $S = \text{chunk} \times E / 4$ 个（$E$ 是元素的字节数）。stride 与数组声明成几维、下标怎么写都无关。

另一个变量是访存指令的位宽。以向量化的方式一次访问 $w$ 个连续的 word，$w$ 取 1、2、4，对应 32 bit、`float2` 的 8 字节、`float4` 的 16 字节。于是线程 $t$ 读的是第 $St$ 到 $St + w - 1$ 个 word。

两条前提成立时，冲突路数可以直接从上面的硬件事实数出来：

1. **$S$ 是整数个 word。** fp16 的 `chunk` 取奇数时它带半个 word，取公约数无从谈起，只能按字节地址逐个算出线程落在哪条 bank 上再数。
2. **32 个线程请求的 $32w$ 个 word 互不相同**，即 $S \ge w$。落在同一个 word 上的线程走广播，不占额外的周期，下面按 $32w$ 个 word 计数就不成立；$S < w$ 时相邻线程的向量区间彼此重叠，同样不成立。

整个 warp 要取 $32w$ 个 word，而 shared memory 每周期最多处理 32 个，所以这条指令至少要 $w$ 个周期 —— 这是位宽带来的下界，与地址无关，冲突指的是超出这个下界的部分。$St \bmod 32$ 只取到 $g = \gcd(S, 32)$ 的倍数，也就是 $32/g$ 条 bank，每条被碰到 $g$ 次；再叠加 $j = 0, \dots, w-1$ 的平移。$g$ 与 $w$ 都是 2 的幂，必有一个整除另一个，于是分两种情形：

| | bank 的落点 | 周期数 | 冲突 |
| --- | --- | --- | --- |
| $g \le w$ | 32 条 bank 各被请求 $w$ 次 | $w$，正好是下界 | 无 |
| $g > w$ | 只有 $(32/g) \cdot w$ 条被碰到，各 $g$ 次 | $g$ | $g / w$ 路 |

$$\text{冲突路数} \ \ge\ \max\left(1,\ \frac{\gcd(S,\ 32)}{w}\right)$$

标量读（$w = 1$）时它就是 $\gcd(S, 32)$：$S$ 与 32 互素时 32 个线程铺满 32 条 bank，无冲突；$S$ 是 32 的倍数时全部挤在同一条 bank 上，32 路串行 —— fp16、`chunk = 64` 就是后者，$S$ 是 128 字节即 32 个 word。

写成不等式，是因为最后一步假定硬件能把任意一组无冲突的 word 凑进一个周期，而 NVIDIA 没有公开 64 bit 与 128 bit 访问时 lane 的实际分组方式。

## 用 pad 改变 stride

stride 由 `chunk` 决定，而 `chunk` 通常由算法决定，往往无法任意改动。当发生$N$路冲突时，一种可行方式是在每段末尾增加 `pad` 个元素，让 stride 变成 `chunk + pad` —— `gcd` 随之改变，冲突路数也就随之改变。例如：fp16、`chunk = 64`时，上只要加 2 个元素，stride 就从 32 个 word 变成 33 个，与 32 互素，冲突完全消失。

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

<figcaption>四张图是同一个 kernel 在四个 pad 取值下，一个 warp 的 32 个线程在 32 条 bank 上的落点。格内数字是落在这条 bank 上的线程数，最大的那个数就是冲突路数。<code>pad = 0</code> 时 32 个线程全挤在 bank 0；加 2 个元素之后 stride 变成 33 个 word，与 32 互素，32 个线程正好铺满 32 条 bank。</figcaption>

</figure>

## pad 该取多少

判据由上面的式子给出，**两条都要满足**：

| 访问位宽 | $w$ | 冲突路数要求 | 段起点的对齐要求 | fp32、`chunk = 64` 上最快的 pad |
| --- | --- | --- | --- | --- |
| 32 bit，标量读 | 1 | $\gcd(S, 32) = 1$ | 4 字节 | `pad = 1`（$S = 65$ word） |
| 64 bit，`float2` | 2 | $\gcd(S, 32) \le 2$ | 8 字节 | `pad = 2`（$S = 66$ word） |
| 128 bit，`float4` | 4 | $\gcd(S, 32) \le 4$ | 16 字节 | `pad = 4`（$S = 68$ word） |

两条方向相反，所以不能只看一条：**位宽越宽，对冲突路数的要求越松，对对齐的要求越紧**。奇数 pad 对标量读是最优解，对向量化读却是最差的 —— 它让段起点错开 4 字节。128 bit 的 shared 读要求 16 字节自然对齐，对齐不足时这条指令的行为是未定义的；编译期能看出对齐不足时，编译器会退回窄指令，下面实测里 $\gcd = 1$ 那一行就是这种情形。

实测（H200，fp32，`chunk = 64`，消费循环重复 32 遍使 shared 一侧成为瓶颈，耗时随遍数成比例）给出每种组合比同位宽最快的那一行慢多少倍：

| $\gcd(S, 32)$ | 32 bit | 64 bit | 128 bit | 下界预测（32 / 64 / 128） |
| --- | --- | --- | --- | --- |
| 1 | **1.00** | 1.10 | 2.27 | 1 / 1 / 1 |
| 2 | 1.72 | **1.00** | 1.01 | 2 / 1 / 1 |
| 4 | 3.29 | 1.81 | **1.00** | 4 / 2 / 1 |
| 8 | 6.52 | 3.55 | 4.04 | 8 / 4 / 2 |
| 16 | 12.97 | 7.62 | 7.99 | 16 / 8 / 4 |
| 32 | 25.82 | 13.97 | 15.71 | 32 / 16 / 8 |

对角线上的三个 1.00 就是上表那三行推荐。$g \le w$ 那一侧下界是紧的；$g > w$ 那一侧下界不紧，128 bit 实测恰好是它的 2 倍（4.04 对 2、7.99 对 4、15.71 对 8）；成因要更低层的证据才能定，NVIDIA 未公开这两种位宽下 lane 的分组方式。$\gcd = 1$ 那一行的 2.27 是对齐造成的：$S = 65$ word 即 260 字节，不是 16 的倍数。

## pad 要按 chunk 算 {#pad-per-chunk}

固定字节数的 pad 会在某些 `chunk` 上落回最坏情形。段起点的对齐要求把候选限制在 16 字节的整数倍，记 pad 为 $k$ 个 16 字节（$k = 1, 2, \dots$），于是

$$S = \frac{\text{chunk} \times E}{4} + 4k \ \text{word}$$

$k$ 固定时 $\gcd(S, 32)$ 随 `chunk` 变。fp16 与 bf16（$E = 2$）的几个 `chunk`：

| chunk | $S$（$k = 1$） | $\gcd(S, 32)$ | $S$（$k = 2$） | $\gcd(S, 32)$ |
| --- | --- | --- | --- | --- |
| 32 | 20 | **4** | 24 | 8 |
| 56 | 32 | 32 | 36 | **4** |
| 64 | 36 | **4** | 40 | 8 |
| 72 | 40 | 8 | 44 | **4** |
| 128 | 68 | **4** | 72 | 8 |

`chunk = 56` 那一行的 $S$ 恰好是 32 个 word，与 `pad = 0` 落在同一条 bank 上 —— pad 加了，冲突没消。做法是在 $k = 1$ 与 $k = 2$ 两个候选里取 $\gcd(S, 32)$ 小的那个。$\text{chunk} \times E$ 是 16 的倍数时，两者必有一个把 $\gcd$ 降到 4：记 $q = \text{chunk} \times E / 16$，则 $S = 4(q + k)$，$\gcd(S, 32) = 4 \gcd(q + k, 8)$，而 $q + 1$ 与 $q + 2$ 一奇一偶。

实测（H200，SM 时钟锁在 1830 MHz，bf16，CUPTI 设备耗时，每次迭代前清 L2，取 200 次的中位数，镜像 `ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-afcebed1-dev`）。四行的线程数都取到让 `chunk` 等于 56，同一行两列只差 pad：

| 输入 | 线程数 × chunk | $k = 1$（pad 8 个元素）<br>TB/s | $k = 2$（pad 16 个元素）<br>TB/s |
| --- | --- | --- | --- |
| $2048 \times 3584$ | 64 × 56 | 1.38 | **3.07**{ .win } |
| $2048 \times 7168$ | 128 × 56 | 1.58 | **3.29**{ .win } |
| $1024 \times 14336$ | 256 × 56 | 1.51 | **3.05**{ .win } |
| $512 \times 28672$ | 512 × 56 | 1.32 | **2.33**{ .win } |

这四个宽度是 Qwen2-7B 的 hidden size、Llama-3-70B 的 hidden size、Llama-3-8B 与 Llama-3-70B 的 FFN 中间维，不是构造出来的反例。

这组测量的 shared 一侧不是标量读：`cuobjdump` 显示 $S = 36$ word 时每线程 16 条 `LDS.64`，即 $w = 2$，所以路数是 $\gcd(S, 32) / 2$ —— 两列分别是 16 路与 2 路。路数差 8 倍而带宽只差 2.2 倍，是因为 2 路那一列的瓶颈已经回到 DRAM。段起点错开 8 字节以下时它退回 `LDS`（$w = 1$）—— 位宽由编译器定，不由声明 pad 的人定，所以上表两列都要实测，不能只算 $\gcd$。

## 实测扫描

下面的实测每线程逐个元素读（$w = 1$），`chunk` 取 2 的幂，每个线程读自己那一段 —— 上面两个前提都成立。

H200，SM 时钟锁在 1830 MHz。fp16，输入 $65536 \times 4096$（512 MB，大于 60 MiB 的 L2）。kernel 把整行搬进 shared memory，每个线程一段 `chunk + pad` 个元素，逐段做串行前缀积再写回，读加写共 1 GB。括号内是上面公式预测的冲突路数：

| chunk | 线程数 | pad = 0<br>TB/s | pad = 2<br>TB/s | pad = 4<br>TB/s | pad = 8<br>TB/s | pad = 16<br>TB/s |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 256 | 1.63（8 路） | 2.99（1 路） | **3.29**{ .win }（2 路） | 2.58（4 路） | 0.86（16 路） |
| 32 | 128 | 0.89（16 路） | 3.32（1 路） | **3.35**{ .win }（2 路） | 2.57（4 路） | 1.54（8 路） |
| 64 | 64 | 0.46（32 路） | 3.63（1 路） | **3.70**{ .win }（2 路） | 2.74（4 路） | 1.62（8 路） |
| 128 | 32 | 0.46（32 路） | 3.25（1 路） | **3.31**{ .win }（2 路） | 2.76（4 路） | 1.61（8 路） |

20 个配置里，1 路与 2 路在 3.0 以上，4 路降到 2.6 附近，8 路及以上跌到 1.6 以下。带宽随预测的路数单调下降，所以在这组条件下公式可以直接用来缩小 pad 的候选。

## 使用时的注意事项

1. **公式给出候选，最终值靠实测。** 上表里 1 路与 2 路都在 3.0 以上，4 路降到 2.6 附近，8 路及以上跌到 1.6 以下，所以公式的用处是把候选缩到「算出来不超过 $w$ 路」的那几个。这几个之间要实测：四组 `chunk` 上 2 路都略高于 1 路，但差距从 0.9% 到 10% 不等，不构成一条可以照搬的规则。

2. **改完 pad 要重扫 `chunk`。** `pad = 0` 那一列最优的是 `chunk = 16`（1.63），`pad = 4` 那一列最优的是 `chunk = 64`（3.70）。前一列的排序主要由冲突路数决定 —— `chunk = 16` 撞的是 8 路，`chunk = 64` 撞的是 32 路。冲突消掉之后四组都是 2 路，最优点换了位置。表里线程数与 `chunk` 联动（两者之积恒为行宽 4096），所以换位置的成因不止 `chunk` 一个，占用率与循环长度也跟着在变；能确定的只是改完 pad 之后 `chunk` 的排序会变。

3. **`chunk` 变了要重算 pad。** pad 写成固定字节数，等于假定 $\gcd(S, 32)$ 与 `chunk` 无关，而[上一节](#pad-per-chunk)那张表说明它不是：`chunk = 56`、$k = 1$ 时 $S$ 回到 32 个 word。把 pad 写成 `chunk` 的函数 —— 在 $k = 1, 2$ 里取 $\gcd(S, 32)$ 小的那个 —— 才和这一页的判据自洽。

4. **这一页只在数据经过 shared memory 时适用。** 按[优化 global memory 访问](global-memory-access.md#coalescing)里的取舍，$V$ 小时用向量化的 blocked，数据直接进寄存器，不经 shared memory，没有 bank 冲突可言；$V$ 大到寄存器压力压低占用率时才改用 staged，以及整行需要被 block 内所有线程共享时，这一页才适用。

下面两段是 shared 缓冲的声明，差别只在 stride。反例，stride 恰好是 32 个 word 的倍数：

```python
sh = T.alloc_shared((threads * chunk,), dtype)          # stride = chunk 个元素
```

正例，pad 按 `chunk` 算，取 16 字节的整数倍里 $\gcd(S, 32)$ 最小的那个：

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
