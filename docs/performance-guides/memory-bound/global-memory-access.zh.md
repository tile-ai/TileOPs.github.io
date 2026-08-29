# 优化 global memory 访问

一个线程要读一行里的多个元素时，写法有四种。这一页给出它们在两个 workload 上的实测对比，以及怎么挑一种。

## 确认 DRAM 带宽是否为当前的瓶颈 {#regime}

[Elementwise](https://tile-ai.github.io/TileOPs.github.io/api/elementwise/) 与 [Reduction](https://tile-ai.github.io/TileOPs.github.io/api/reduction/) 是典型的访存受限 kernel。下面每条建议都写明触发的条件、成因，以及反例与正例代码。

这一页的实测都在同一组条件下取得：**输入大于 L2 的 60 MiB，且 block 数足以填满整卡**（H200 有 132 个 SM）。此时 DRAM 带宽是主要瓶颈，访存模式的差别直接反映在性能上。

!!! warning "适用范围"

    这组条件之外，主要性能瓶颈可能由别的因素决定，这里给出的一些结论会反转。

下表按这两个条件划出三个区间，逐行给出判据，以及这两页的结论在各区间里怎么用。表里的瓶颈是主导因素，kernel 越复杂，同时起作用的因素越多：

| 区间 | 判据 | 主要瓶颈 | 结论的用法 |
| --- | --- | --- | --- |
| 带宽饱和 | 输入 > 60 MiB，block 数在 SM 数的两倍以上 | DRAM 带宽，即 sector 利用率 | 直接适用 |
| 数据小 | 输入装得进 L2，单次耗时在几十微秒以内 | kernel 发射的固定开销、缓存状态 | 避开反例即可，换 access pattern 没有收益 |
| block 少 | block 数不到 SM 数的两倍 | 每条载入指令的宽度、在飞的字节数 | 保住载入宽度优先，改动逐个实测 |

- **数据小时，发射开销与缓存状态占主导。** 同一个行求和 kernel 的四种 access pattern（fp16，256 线程，时钟未锁）在 65536 × 4096（512 MB）上测得的访存带宽是 4.20 到 4.43 TB/s，彼此相差不超过 6%；换成 2048 × 4096（16 MB，装得进 L2）后单次耗时十几微秒，同一个 access pattern 两次测量之间可差三倍。这个区间里换 access pattern 没有收益，制约性能的是别的因素。
- **block 少时，载入宽度带来的收益大于合并规则算出的差别。** warp 数量不足，靠并发的请求数掩盖访存延迟不再可行，只能让每个请求更宽、每个线程持有更多在飞的字节。以载入宽度换取其他好处的改法，在这个区间都可能反转。

## 合并 global memory 的访存 {#coalescing}

**一次 global memory 读取的数据访问单位是 32 字节的 sector。**

| 名称 | 大小 | 是什么 |
| --- | --- | --- |
| cache line | 128 字节 | L1 与 L2 的缓存行，也是缓存查找的单位 |
| **sector** | **32 字节** | 一条 cache line 由 4 个 sector 组成，L1 与 L2 之间按 sector 传输 |

缓存查找时以 cache line 为单位，搬运数据时以 sector 为单位：某个 sector 未命中，L1 就只向 L2 请求这一个 sector，不必把整条 cache line 都拉过来。由此得到的结论是**取 1 个字节和取满 32 个字节的代价相同**。于是一条访存指令的好坏由 **sector 利用率**衡量：`真正用到的字节 / (覆盖的 sector 数 × 32)`。

硬件把一个 warp 的 32 个访问合并成尽可能少的 32 字节事务。事务数最少要同时满足三个因素：

1. **地址连续** —— 同一条指令里 32 个线程的地址首尾相接，不留空洞；
2. **按 32 字节对齐** —— 起始地址是 32 的倍数，一段数据不会多占一个 sector；
3. **每个线程一次取满 16 字节** —— 一条指令覆盖 $32 \times 16 = 512$ 个连续字节，即 16 个满载的 sector。

三个因素同时成立时，这条访存指令对硬件最友好。

一个线程要读 $V$ 个元素时（$V$ = 一行的元素数 / 线程数），有四种 access pattern。

**blocked** —— 每个线程负责一段连续的元素。固定 `c` 时相邻线程的地址相隔 $V$ 个元素，违反第一个因素，sector 利用率是 $1/V$：

```python
for c in T.serial(V):
    acc[0] = acc[0] * X[row, tx * V + c]
```

**striped** —— 相邻线程取相邻元素。地址连续了，但每个线程一次只取一个元素，违反第三个因素，$V$ 个元素要发 $V$ 条指令：

```python
for c in T.serial(V):
    acc[0] = acc[0] * X[row, c * threads + tx]
```

**blocked + 向量化** —— 仍是每线程一段连续的，但改用 `T.vectorized` 一次读满 16 字节，三个因素全部满足：

```python
buf = T.alloc_local((V,), dtype)

for c in T.vectorized(V):
    buf[c] = X[row, tx * V + c]
for c in T.serial(V):
    acc[0] = acc[0] * buf[c]
```

**staged** —— 搬运交给 `T.Parallel`，消费改从 shared memory 读，同样满足三个因素：

```python
sh = T.alloc_shared((threads, V + pad), dtype)

for t, c in T.Parallel(threads, V):
    sh[t, c] = X[row, t * V + c]
T.sync_threads()
for c in T.serial(V):
    acc[0] = acc[0] * sh[tx, c]
```

四种 access pattern 的差别在于：**「哪个线程读哪些元素、一次读多宽」这个映射由谁决定。**

`T.serial` 的语义是循环体由单个线程顺序执行，索引表达式被逐字翻译成访存指令，不做合并也不做向量化 —— 编程者写出的模式就是硬件看到的模式。

`T.vectorized`、`T.Parallel`、`T.copy` 则由 TileLang 的 **layout inference** 决定，三者的差别在于编程者还需要写明多少：`T.vectorized` 要写明每线程一次访问的宽度，线程映射由 layout inference 推导；`T.Parallel` 连宽度也不必写，循环维度怎么分给线程、一次读多宽都由它决定；`T.copy` 只写源和目标两个区域，整段搬运由它生成（需要接管推导结果时，另有 `coalesced_width` 与 `loop_layout` 两个参数）。剩下的向量化、地址对齐、以及在 shared memory 一侧避开 bank 冲突，都由 layout inference 负责 —— 这些正是对硬件友好但手写容易出错的部分。

**用 `T.serial` 手写下标时，上面三个因素要自己逐一保证；交给 layout inference 时，只需写明搬运的范围。** 三者之间怎么选、各自能跑到多少带宽，见下面的实测。

<figure class="access-patterns" markdown="1">

<svg class="tf-access" viewBox="0 0 520 352" role="img" aria-label="三种访存模式下，一个 warp 的一条读取指令触及的元素，以及硬件因此取回的 sector。blocked 覆盖 4 个 sector，每个只用到 2 个元素；向量化后的 blocked 与 striped 都完全合并；staged 的搬运阶段与 striped 一样合并，消费阶段在 shared memory 上，不存在 sector。">
<text class="ap-title" x="0" y="38.0">blocked</text>
<text class="ap-sub" x="0" y="52.0">tx * V + c</text>
<rect class="ap-sector ap-sector--fetched" x="136.0" y="28.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="229.0" y="28.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="322.0" y="28.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="415.0" y="28.0" width="88.0" height="22.0" rx="2"/>
<text class="ap-owner" x="141.5" y="23.0" text-anchor="middle">0</text>
<text class="ap-owner" x="185.5" y="23.0" text-anchor="middle">1</text>
<text class="ap-owner" x="234.5" y="23.0" text-anchor="middle">2</text>
<text class="ap-owner" x="278.5" y="23.0" text-anchor="middle">3</text>
<text class="ap-owner" x="327.5" y="23.0" text-anchor="middle">4</text>
<text class="ap-owner" x="371.5" y="23.0" text-anchor="middle">5</text>
<text class="ap-owner" x="420.5" y="23.0" text-anchor="middle">6</text>
<text class="ap-owner" x="464.5" y="23.0" text-anchor="middle">7</text>
<rect class="ap-cell ap-cell--t0" x="136.0" y="28.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="141.5" cy="39.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="147.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="158.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="169.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="180.0" y="28.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="185.5" cy="39.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="191.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="202.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="213.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="229.0" y="28.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="234.5" cy="39.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="240.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="251.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="262.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="273.0" y="28.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="278.5" cy="39.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="284.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="295.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="306.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="322.0" y="28.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="327.5" cy="39.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="333.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="344.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="355.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="366.0" y="28.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="371.5" cy="39.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="377.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="388.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="399.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="415.0" y="28.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="420.5" cy="39.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="426.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="437.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="448.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="459.0" y="28.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="464.5" cy="39.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="470.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="481.0" y="28.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="492.0" y="28.0" width="11.0" height="22.0"/>
<text class="ap-note" x="136.0" y="65.0">取回 4 个 sector，每个只用到 2 / 8 个元素</text>
<text class="ap-title" x="0" y="108.0">blocked + 向量化</text>
<text class="ap-sub" x="0" y="122.0">T.vectorized(V)</text>
<rect class="ap-sector ap-sector--fetched" x="136.0" y="98.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="229.0" y="98.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="322.0" y="98.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="415.0" y="98.0" width="88.0" height="22.0" rx="2"/>
<text class="ap-owner" x="141.5" y="93.0" text-anchor="middle">0</text>
<text class="ap-owner" x="185.5" y="93.0" text-anchor="middle">1</text>
<text class="ap-owner" x="234.5" y="93.0" text-anchor="middle">2</text>
<text class="ap-owner" x="278.5" y="93.0" text-anchor="middle">3</text>
<text class="ap-owner" x="327.5" y="93.0" text-anchor="middle">4</text>
<text class="ap-owner" x="371.5" y="93.0" text-anchor="middle">5</text>
<text class="ap-owner" x="420.5" y="93.0" text-anchor="middle">6</text>
<text class="ap-owner" x="464.5" y="93.0" text-anchor="middle">7</text>
<rect class="ap-cell ap-cell--t0" x="136.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="141.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="147.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="152.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="158.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="163.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="169.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="174.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="180.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="185.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="191.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="196.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="202.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="207.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="213.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="218.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="229.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="234.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="240.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="245.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="251.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="256.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="262.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="267.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="273.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="278.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="284.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="289.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="295.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="300.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="306.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="311.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="322.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="327.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="333.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="338.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="344.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="349.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="355.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="360.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="366.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="371.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="377.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="382.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="388.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="393.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="399.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="404.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="415.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="420.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="426.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="431.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="437.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="442.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="448.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="453.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="459.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="464.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="470.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="475.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="481.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="486.5" cy="109.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="492.0" y="98.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="497.5" cy="109.0" r="3.2"/>
<text class="ap-note" x="136.0" y="135.0">一条 16 字节向量读，取回 4 个 sector，全部用满</text>
<text class="ap-title" x="0" y="182.0">striped</text>
<text class="ap-sub" x="0" y="196.0">c * threads + tx</text>
<rect class="ap-sector ap-sector--fetched" x="136.0" y="172.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector" x="229.0" y="172.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector" x="322.0" y="172.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector" x="415.0" y="172.0" width="88.0" height="22.0" rx="2"/>
<text class="ap-owner" x="141.5" y="167.0" text-anchor="middle">0</text>
<text class="ap-owner" x="152.5" y="167.0" text-anchor="middle">1</text>
<text class="ap-owner" x="163.5" y="167.0" text-anchor="middle">2</text>
<text class="ap-owner" x="174.5" y="167.0" text-anchor="middle">3</text>
<text class="ap-owner" x="185.5" y="167.0" text-anchor="middle">4</text>
<text class="ap-owner" x="196.5" y="167.0" text-anchor="middle">5</text>
<text class="ap-owner" x="207.5" y="167.0" text-anchor="middle">6</text>
<text class="ap-owner" x="218.5" y="167.0" text-anchor="middle">7</text>
<rect class="ap-cell ap-cell--t0" x="136.0" y="172.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="141.5" cy="183.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="147.0" y="172.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="152.5" cy="183.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="158.0" y="172.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="163.5" cy="183.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="169.0" y="172.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="174.5" cy="183.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="180.0" y="172.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="185.5" cy="183.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="191.0" y="172.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="196.5" cy="183.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="202.0" y="172.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="207.5" cy="183.0" r="3.2"/>
<rect class="ap-cell ap-cell--t1" x="213.0" y="172.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="218.5" cy="183.0" r="3.2"/>
<rect class="ap-cell ap-cell--t0" x="229.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="240.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="251.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="262.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="273.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="284.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="295.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="306.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="322.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="333.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="344.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="355.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="366.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="377.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="388.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="399.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="415.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="426.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="437.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="448.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="459.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="470.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t0" x="481.0" y="172.0" width="11.0" height="22.0"/>
<rect class="ap-cell ap-cell--t1" x="492.0" y="172.0" width="11.0" height="22.0"/>
<text class="ap-note" x="136.0" y="209.0">取回 1 个 sector，8 / 8 个元素全部用到</text>
<text class="ap-title" x="0" y="256.0">staged</text>
<text class="ap-sub" x="0" y="270.0">T.Parallel 整行搬运</text>
<rect class="ap-sector ap-sector--fetched" x="136.0" y="246.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="229.0" y="246.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="322.0" y="246.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--fetched" x="415.0" y="246.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-cell" x="136.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="141.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="147.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="152.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="158.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="163.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="169.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="174.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="180.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="185.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="191.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="196.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="202.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="207.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="213.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="218.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="229.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="234.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="240.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="245.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="251.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="256.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="262.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="267.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="273.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="278.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="284.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="289.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="295.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="300.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="306.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="311.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="322.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="327.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="333.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="338.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="344.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="349.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="355.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="360.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="366.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="371.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="377.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="382.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="388.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="393.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="399.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="404.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="415.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="420.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="426.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="431.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="437.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="442.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="448.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="453.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="459.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="464.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="470.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="475.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="481.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="486.5" cy="257.0" r="3.2"/>
<rect class="ap-cell" x="492.0" y="246.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="497.5" cy="257.0" r="3.2"/>
<text class="ap-note" x="136.0" y="283.0">取回 4 个 sector，全部用满</text>
<text class="ap-sub" x="0" y="318.0">staged[tx * V + c]</text>
<rect class="ap-sector ap-sector--onchip" x="136.0" y="294.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--onchip" x="229.0" y="294.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--onchip" x="322.0" y="294.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-sector ap-sector--onchip" x="415.0" y="294.0" width="88.0" height="22.0" rx="2"/>
<rect class="ap-cell" x="136.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="141.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="147.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="152.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="158.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="163.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="169.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="174.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="180.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="185.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="191.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="196.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="202.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="207.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="213.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="218.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="229.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="234.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="240.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="245.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="251.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="256.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="262.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="267.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="273.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="278.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="284.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="289.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="295.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="300.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="306.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="311.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="322.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="327.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="333.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="338.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="344.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="349.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="355.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="360.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="366.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="371.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="377.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="382.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="388.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="393.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="399.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="404.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="415.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="420.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="426.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="431.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="437.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="442.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="448.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="453.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="459.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="464.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="470.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="475.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="481.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="486.5" cy="305.0" r="3.2"/>
<rect class="ap-cell" x="492.0" y="294.0" width="11.0" height="22.0"/>
<circle class="ap-dot" cx="497.5" cy="305.0" r="3.2"/>
<text class="ap-note" x="136.0" y="331.0">在 shared memory 上消费，不存在 sector</text>
<text class="ap-scale" x="0" y="348.0">示意图：fp32、8 个线程、V = 4，一个 sector 装 8 个元素。正文实测用 bf16、256 线程、V = 16，形状相同。</text>
</svg>

<figcaption>一条读取指令。紫点是各线程在这条指令里读到的元素，青色底是硬件因此取回的 sector；格子上方是线程号，深浅交替标出线程边界。最后一行的紫色虚线框是 shared memory，那里不按 sector 组织。</figcaption>

</figure>

## 实测对比

我们对两个 workload 在 H200 上进行实测，比较上面四种 access pattern 各自能跑到多少**访存带宽**（搬运的字节数除以 kernel 耗时，单位 TB/s），这两个 workload 的计算对元素的处理顺序有不同要求 —— 这个要求会决定哪几种 access pattern 可用。

测试中 SM 时钟锁在 1830 MHz；输入 bf16 的 $65536 \times 4096$（512 MB，**必须大于 L2 的 60 MiB**，否则测到的是 L2 带宽）；每个配置跑三次，三次的结果一致到 ±0.5%。staged 在表里占两列：一列不加 pad（此时 stride 恰好是 $V$ 个 word，产生 bank 冲突，见[优化 shared memory 访问](shared-memory-access.md)），一列是在若干个 pad 取值中测到的最优值。

**workload 1，按行求乘积**：一行的元素相乘，先后顺序不影响结果，每行只输出一个值，四种 access pattern 都可用。

| 线程数 | $V$ | blocked<br>TB/s | striped<br>TB/s | blocked + 向量化<br>TB/s | staged 不加 pad<br>TB/s | staged 加 pad<br>TB/s |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 8 | 3.02 | 3.04 | **3.22**{ .win } | 3.05 | 3.02 |
| 256 | 16 | 1.83 | 3.31 | **3.81**{ .win } | 3.79 | 3.79 |
| 128 | 32 | 0.95 | 3.36 | **3.99**{ .win } | 3.81 | 3.98 |
| 64 | 64 | 0.48 | 3.49 | 3.32 | 2.80 | **4.02**{ .win } |
| 32 | 128 | 0.47 | 3.43 | 3.11 | 2.83 | **3.88**{ .win } |

**workload 2，按行求串行前缀积**：每个位置的结果依赖它左边的全部元素，顺序不能变，整行都要写回。striped 在这里不可用（线程无法持有连续的一段）。

| 线程数 | $V$ | blocked<br>TB/s | blocked + 向量化<br>TB/s | staged 不加 pad<br>TB/s | staged 加 pad<br>TB/s |
| --- | --- | --- | --- | --- | --- |
| 512 | 8 | 0.92 | **4.20**{ .win } | 2.47 | 3.15 |
| 256 | 16 | 0.42 | **3.85**{ .win } | 1.63 | 3.29 |
| 128 | 32 | 0.34 | 2.76 | 0.89 | **3.35**{ .win } |
| 64 | 64 | 0.26 | 1.80 | 0.46 | **3.69**{ .win } |
| 32 | 128 | 0.27 | 1.60 | 0.46 | **3.24**{ .win } |

## access pattern 的取舍

1. **逐元素的 blocked 在 $V > 1$ 时总是最差的 access pattern。** 固定 `c` 时相邻线程的地址相隔 $V$ 个元素，sector 利用率是 $1/V$，所以 $V$ 越大越差 —— workload 1 的表里从 $V = 8$ 的 3.02 掉到 $V = 64$ 的 0.48。这个关系由访存合并的规则决定，不随形状改变。

2. **$V$ 小时用向量化的 blocked；$V$ 大到寄存器压力压低占用率时，改用加了 pad 的 staged。** 向量化把整段留在寄存器里（bf16 是每线程 $V/2$ 个），staged 把它放进 shared memory，用一次同步换回寄存器。翻转点取决于 kernel 里其余部分还剩多少寄存器预算，不是一个固定的 $V$：上面两个 workload 在同一个行宽下就分别落在 $V = 64$ 与 $V = 32$。**这个翻转点要在自己的 kernel 上测。**

3. **staged 的 shared 缓冲要避开 bank 冲突。** 声明成 `(threads, V)` 时 stride 恰好是 $V$ 个 word，$V$ 为 2 的幂就一定产生冲突；pad 的算法与候选见[优化 shared memory 访问](shared-memory-access.md#pad-per-chunk)。workload 2 的表里，同一个配置不加 pad 是 0.46，加 pad 是 3.69。

4. **striped 完全合并，但每个元素要发一条指令。** 所以它好于逐元素的 blocked、差于向量化的 blocked（$V = 16$ 上 3.31 对 1.83 与 3.81），适合改动量比最后一点带宽更重要的场合。它让线程持有的元素不连续，因此要求线程持有连续一段的计算（例如串行前缀）用不了它。

下面两段是推荐 access pattern 的完整模板，`M`、`N`、`V`、`threads`、`pad`、`dtype` 都是编译期常量。本页开头的四段代码里，逐元素的 blocked 是反例，不要照抄；striped 可用但不是最快的一种（见上面取舍的第 4 条）。

**推荐的 access pattern，小 $V$** —— 向量化的 blocked：

```python
@T.prim_func
def main(X: T.Tensor((M, N), dtype), Out: T.Tensor((M, threads), "float32")):
    with T.Kernel(M, threads=threads) as row:
        tx = T.get_thread_binding()
        buf = T.alloc_local((V,), dtype)
        acc = T.alloc_local((1,), "float32")
        acc[0] = T.cast(1.0, "float32")

        for c in T.vectorized(V):                  # 一条 16 字节向量读
            buf[c] = X[row, tx * V + c]

        for c in T.serial(V):                      # 消费，顺序随计算需要
            acc[0] = acc[0] * T.cast(buf[c], "float32")

        Out[row, tx] = acc[0]
```

**推荐的 access pattern，大 $V$** —— 加了 pad 的 staged（翻转点见上面取舍的第 2 条）：

```python
@T.prim_func
def main(X: T.Tensor((M, N), dtype), Out: T.Tensor((M, threads), "float32")):
    with T.Kernel(M, threads=threads) as row:
        tx = T.get_thread_binding()
        sh = T.alloc_shared((threads, V + pad), dtype)   # pad 的选法见 shared memory 那一篇
        acc = T.alloc_local((1,), "float32")
        acc[0] = T.cast(1.0, "float32")

        for t, c in T.Parallel(threads, V):        # 搬运，映射由 layout inference 给出
            sh[t, c] = X[row, t * V + c]
        T.sync_threads()

        for c in T.serial(V):                      # 消费
            acc[0] = acc[0] * T.cast(sh[tx, c], "float32")

        Out[row, tx] = acc[0]
```
