# Optimizing Global Memory Access

A thread reading several elements of a row can be written four ways. This page
measures all four on two workloads and says how to pick one.

## Checking whether DRAM bandwidth is the current limit {#regime}

[Elementwise](https://tile-ai.github.io/TileOPs.github.io/api/elementwise/) and
[Reduction](https://tile-ai.github.io/TileOPs.github.io/api/reduction/) are the
typical memory-bound kernels. Each recommendation below states the condition
that triggers it, the cause, and code for both the wrong and the right form.

Every measurement on this page was taken under one set of conditions: **an input
larger than the 60 MiB L2, and enough blocks to fill the whole card** (an H200
has 132 SMs). DRAM bandwidth is the main limit there, and a difference in access
pattern shows up directly in performance.

!!! warning "Where this applies"

    Outside these conditions something else may set the limit, and some of the
    conclusions here reverse.

The table below splits the space into three regimes along those two conditions,
giving the test for each and how the conclusions of these two pages apply in it.
The limit named is the dominant one; the more complex the kernel, the more
factors act at once:

| Regime | Test | Main limit | How to use the conclusions |
| --- | --- | --- | --- |
| Bandwidth saturated | Input > 60 MiB, blocks at more than twice the SM count | DRAM bandwidth, that is sector utilization | Apply directly |
| Small data | Input fits in L2, one call takes tens of microseconds or less | Fixed launch overhead, cache state | Avoid the wrong forms; changing access pattern buys nothing |
| Few blocks | Blocks fewer than twice the SM count | The width of each load instruction, bytes in flight | Keep load width first, and measure every change |

- **With small data, launch overhead and cache state dominate.** The four access
  patterns of one row-reduction kernel (fp16, 256 threads, clocks unlocked)
  measure 4.20 to 4.43 TB/s on 65536 × 4096 (512 MB), within 6% of each other;
  at 2048 × 4096 (16 MB, which fits in L2) one call takes a dozen or so
  microseconds and two measurements of the *same* access pattern can differ
  threefold. Changing access pattern buys nothing in this regime, because
  something else is setting the pace.
- **With few blocks, load width is worth more than the coalescing rules
  predict.** There are too few warps to hide memory latency behind concurrent
  requests, so the only lever left is making each request wider and each thread
  hold more bytes in flight. Any change that trades load width for something
  else can reverse here.

## Coalescing global memory accesses {#coalescing}

**A global memory read moves data in units of a 32-byte sector.**

| Name | Size | What it is |
| --- | --- | --- |
| cache line | 128 bytes | The L1 and L2 line, and the unit a cache lookup uses |
| **sector** | **32 bytes** | A cache line is 4 sectors; L1 and L2 transfer whole sectors |

Lookup works in cache lines, transfer works in sectors: on a sector miss L1
requests just that sector from L2 rather than pulling the whole line. Which
means **fetching 1 byte costs the same as fetching all 32**. The quality of a
memory instruction is therefore measured by its **sector utilization**:
`bytes actually used / (sectors touched × 32)`.

The hardware coalesces a warp's 32 accesses into as few 32-byte transactions as
it can. Reaching the minimum requires three things at once:

1. **Contiguous addresses** — the 32 threads of one instruction address a run
   with no holes in it;
2. **32-byte alignment** — the start address is a multiple of 32, so a run does
   not spill into one more sector;
3. **16 bytes fetched per thread per instruction** — one instruction then covers
   $32 \times 16 = 512$ contiguous bytes, which is 16 fully used sectors.

An instruction that satisfies all three is as friendly to the hardware as it gets.

A thread reading $V$ elements ($V$ = elements per row / threads) has four access
patterns available.

**blocked** — each thread takes one contiguous run. For a fixed `c`, adjacent
threads are $V$ elements apart, which breaks the first requirement; sector
utilization is $1/V$:

```python
for c in T.serial(V):
    acc[0] = acc[0] * X[row, tx * V + c]
```

**striped** — adjacent threads take adjacent elements. Addresses are contiguous
now, but each thread fetches a single element per instruction, breaking the
third requirement: $V$ elements take $V$ instructions.

```python
for c in T.serial(V):
    acc[0] = acc[0] * X[row, c * threads + tx]
```

**blocked + vectorized** — still one contiguous run per thread, but
`T.vectorized` reads a full 16 bytes at a time, satisfying all three:

```python
buf = T.alloc_local((V,), dtype)

for c in T.vectorized(V):
    buf[c] = X[row, tx * V + c]
for c in T.serial(V):
    acc[0] = acc[0] * buf[c]
```

**staged** — `T.Parallel` performs the copy and consumption reads shared memory,
which also satisfies all three:

```python
sh = T.alloc_shared((threads, V + pad), dtype)

for t, c in T.Parallel(threads, V):
    sh[t, c] = X[row, t * V + c]
T.sync_threads()
for c in T.serial(V):
    acc[0] = acc[0] * sh[tx, c]
```

What separates the four is this: **who decides which thread reads which elements,
and how wide each read is.**

`T.serial` means the loop body runs sequentially on a single thread, and the
index expression is translated literally into memory instructions — no
coalescing, no vectorization. The pattern written is the pattern the hardware
sees.

`T.vectorized`, `T.Parallel`, and `T.copy` leave that to TileLang's **layout
inference**, and differ in how much is still written by hand: `T.vectorized`
takes the width of one access per thread and infers the thread mapping;
`T.Parallel` takes neither, deciding both how loop dimensions are split across
threads and how wide each read is; `T.copy` takes only a source and a
destination region and generates the whole copy (`coalesced_width` and
`loop_layout` are there for taking the inferred result back over). Vectorization,
address alignment, and avoiding bank conflicts on the shared memory side are all
left to layout inference — exactly the parts that are friendly to the hardware
and easy to get wrong by hand.

**Writing indices by hand with `T.serial` means guaranteeing those three
requirements yourself; handing the copy to layout inference means writing only
its extent.** Which to choose, and what bandwidth each reaches, is measured
below.

<figure class="access-patterns" markdown="1">

<svg class="tf-access" viewBox="0 0 520 366" role="img" aria-label="The elements one read instruction of a warp touches under each access pattern, and the sectors the hardware fetches as a result. blocked spans 4 sectors and uses 2 elements of each; vectorized blocked and striped are both fully coalesced; the staging copy of staged coalesces like striped, and its consumption reads shared memory, where there are no sectors.">
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
<text class="ap-note" x="136.0" y="65.0">4 sectors fetched, 2 of 8 elements used in each</text>
<text class="ap-title" x="0" y="100.0">blocked +</text>
<text class="ap-title" x="0" y="115.0">vectorized</text>
<text class="ap-sub" x="0" y="130.0">T.vectorized(V)</text>
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
<text class="ap-note" x="136.0" y="135.0">one 16-byte vector read: 4 sectors fetched, all fully used</text>
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
<text class="ap-note" x="136.0" y="209.0">1 sector fetched, all 8 of 8 elements used</text>
<text class="ap-title" x="0" y="256.0">staged</text>
<text class="ap-sub" x="0" y="270.0">T.Parallel row copy</text>
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
<text class="ap-note" x="136.0" y="283.0">4 sectors fetched, all fully used</text>
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
<text class="ap-note" x="136.0" y="331.0">consumed in shared memory, where there are no sectors</text>
<text class="ap-scale" x="0" y="348.0">Diagram: fp32, 8 threads, V = 4, 8 elements to a sector.</text>
<text class="ap-scale" x="0" y="362.0">The measurements below use bf16, 256 threads, V = 16 — the same shape.</text>
</svg>

<figcaption>One read instruction. A violet dot is an element that thread reads in this instruction, and the teal ground is a sector the hardware fetches because of it; thread numbers sit above the cells, with alternating shades marking thread boundaries. The dashed violet row at the bottom is shared memory, which is not organized in sectors.</figcaption>

</figure>

## Measurements

We measure both workloads on an H200, comparing what **memory bandwidth** each
of the four access patterns reaches (bytes moved divided by kernel time, in
TB/s). The two workloads place different demands on the order in which elements
are processed — and that demand decides which access patterns are available.

The SM clock is locked at 1830 MHz; the input is bf16 $65536 \times 4096$
(512 MB, which **has to exceed the 60 MiB L2**, or the number measured is L2
bandwidth). Each configuration runs three times, and the three agree to within
±0.5%. staged takes two columns: one without padding (where the stride is
exactly $V$ words, which produces bank conflicts — see
[Optimizing Shared Memory Access](shared-memory-access.md)) and one at the best
of several padding values measured.

**Workload 1, product along a row**: the elements of a row are multiplied
together, order does not affect the result, and each row writes one value. All
four access patterns are available.

| Threads | $V$ | blocked<br>TB/s | striped<br>TB/s | blocked + vectorized<br>TB/s | staged, no pad<br>TB/s | staged, padded<br>TB/s |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 8 | 3.02 | 3.04 | **3.22**{ .win } | 3.05 | 3.02 |
| 256 | 16 | 1.83 | 3.31 | **3.81**{ .win } | 3.79 | 3.79 |
| 128 | 32 | 0.95 | 3.36 | **3.99**{ .win } | 3.81 | 3.98 |
| 64 | 64 | 0.48 | 3.49 | 3.32 | 2.80 | **4.02**{ .win } |
| 32 | 128 | 0.47 | 3.43 | 3.11 | 2.83 | **3.88**{ .win } |

**Workload 2, serial prefix product along a row**: each position depends on
every element to its left, the order cannot change, and the whole row is written
back. striped is unavailable here, because a thread cannot hold a contiguous run.

| Threads | $V$ | blocked<br>TB/s | blocked + vectorized<br>TB/s | staged, no pad<br>TB/s | staged, padded<br>TB/s |
| --- | --- | --- | --- | --- | --- |
| 512 | 8 | 0.92 | **4.20**{ .win } | 2.47 | 3.15 |
| 256 | 16 | 0.42 | **3.85**{ .win } | 1.63 | 3.29 |
| 128 | 32 | 0.34 | 2.76 | 0.89 | **3.35**{ .win } |
| 64 | 64 | 0.26 | 1.80 | 0.46 | **3.69**{ .win } |
| 32 | 128 | 0.27 | 1.60 | 0.46 | **3.24**{ .win } |

## Choosing an access pattern

1. **Element-by-element blocked is the worst access pattern whenever $V > 1$.**
   For a fixed `c`, adjacent threads are $V$ elements apart and sector
   utilization is $1/V$, so it gets worse as $V$ grows — workload 1 falls from
   3.02 at $V = 8$ to 0.48 at $V = 64$. That relation follows from the
   coalescing rules and does not change with shape.

2. **Use vectorized blocked at small $V$; once $V$ grows enough that register
   pressure cuts occupancy, switch to padded staged.** Vectorizing keeps the run
   in registers ($V/2$ of them per thread for bf16); staged puts it in shared
   memory, buying the registers back for one synchronization. The crossover
   depends on how much register budget the rest of the kernel leaves, not on a
   fixed $V$: the two workloads above cross at $V = 64$ and $V = 32$ at the same
   row width. **Measure that crossover on your own kernel.**

3. **The shared buffer of staged has to avoid bank conflicts.** Declared as
   `(threads, V)` the stride is exactly $V$ words, which conflicts whenever $V$
   is a power of two; the padding rule and its candidates are in
   [Optimizing Shared Memory Access](shared-memory-access.md#pad-per-chunk).
   In workload 2 the same configuration measures 0.46 unpadded and 3.69 padded.

4. **striped coalesces fully, but spends one instruction per element.** That
   puts it above element-by-element blocked and below vectorized blocked (3.31
   against 1.83 and 3.81 at $V = 16$), which suits cases where the size of the
   change matters more than the last bit of bandwidth. It leaves each thread
   holding non-contiguous elements, so a computation that needs a thread to hold
   a contiguous run — a serial prefix, for one — cannot use it.

The two listings below are complete templates for the recommended access
patterns, where `M`, `N`, `V`, `threads`, `pad`, and `dtype` are all
compile-time constants. Of the four listings at the top of this page,
element-by-element blocked is the wrong form and should not be copied; striped
works but is not the fastest (see point 4 above).

**Recommended at small $V$** — vectorized blocked:

```python
@T.prim_func
def main(X: T.Tensor((M, N), dtype), Out: T.Tensor((M, threads), "float32")):
    with T.Kernel(M, threads=threads) as row:
        tx = T.get_thread_binding()
        buf = T.alloc_local((V,), dtype)
        acc = T.alloc_local((1,), "float32")
        acc[0] = T.cast(1.0, "float32")

        for c in T.vectorized(V):                  # one 16-byte vector read
            buf[c] = X[row, tx * V + c]

        for c in T.serial(V):                      # consume, in whatever order the computation needs
            acc[0] = acc[0] * T.cast(buf[c], "float32")

        Out[row, tx] = acc[0]
```

**Recommended at large $V$** — padded staged (for the crossover, see point 2
above):

```python
@T.prim_func
def main(X: T.Tensor((M, N), dtype), Out: T.Tensor((M, threads), "float32")):
    with T.Kernel(M, threads=threads) as row:
        tx = T.get_thread_binding()
        sh = T.alloc_shared((threads, V + pad), dtype)   # for pad, see the shared memory page
        acc = T.alloc_local((1,), "float32")
        acc[0] = T.cast(1.0, "float32")

        for t, c in T.Parallel(threads, V):        # copy; the mapping comes from layout inference
            sh[t, c] = X[row, t * V + c]
        T.sync_threads()

        for c in T.serial(V):                      # consume
            acc[0] = acc[0] * T.cast(sh[tx, c], "float32")

        Out[row, tx] = acc[0]
```
