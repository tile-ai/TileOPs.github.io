# Optimizing Shared Memory Access

Routing data through shared memory adds another access pattern to consider.
Writing data from global memory into shared memory and reading it from shared
memory into registers both access shared memory. This page covers bank conflicts
in those two steps: what determines them, how padding removes them, and what the
padding should be computed from.

Every measurement on this page was taken on an H200 with the SM clock locked at
1830 MHz, an input larger than the 60 MiB L2, and enough blocks to fill the
whole card. Outside those conditions, the conclusions can reverse; the tests are
in [Optimizing Global Memory Access](global-memory-access.md#regime).

## The bank structure of shared memory {#bank-conflict}

Shared memory is built from 32 banks, each 4 bytes wide. If its address space is
divided into 4-byte **words** (all counts below are in words), the bank for an
address is `(byte address / 4) mod 32`. A bank serves one word per cycle. When
several threads access shared memory concurrently and land on different banks,
all of those accesses complete in the same cycle. When several threads land on
the same bank, there are three cases:

1. **Different words** — the hardware splits the request into several
   conflict-free requests and serves them one after another. The number of
   splits is the **conflict degree**.
2. **Reading the same word** — any two threads landing inside one word, even on
   different bytes of it, receive that word by broadcast, with no conflict.
   Several broadcasts on different banks are further merged into one multicast.
3. **Writing the same address** — one write takes effect, but which write is
   undefined.

A shared-memory access pattern should be conflict-free whenever possible.

## What determines the conflict degree

Consider a case that is general enough to reason from. The one-dimensional array
`sh` below lives in shared memory, and each thread reads `chunk` contiguous
elements from it:

```python
sh = T.alloc_shared((threads * chunk,), dtype)   # one dimension, threads * chunk elements

for c in T.serial(chunk):
    acc[0] = acc[0] * sh[tx * chunk + c]         # thread tx reads its own run
```

The 32 threads of a warp execute this loop in lockstep. Within one iteration,
`c` has the same value for all of them, while `tx` runs from 0 to 31. The 32
threads therefore access shared memory at a fixed spacing. At `chunk = 64`, one
iteration reads elements `c`, `64 + c`, `128 + c`, …, `1984 + c`. Adjacent
threads are `chunk` elements apart. That difference is the **stride** of the
access, written $S$; in words, $S = \text{chunk} \times E / 4$, where $E$ is the
element size in bytes. The stride is independent of how many dimensions the
array declaration uses and how the index is written.

The other variable is the width of the memory instruction. A vectorized access
reads $w$ contiguous words at a time, where $w$ is 1, 2, or 4: 32 bit, the
8 bytes of a `float2`, or the 16 bytes of a `float4`. Thread $t$ then reads
words $St$ through $St + w - 1$.

Under two premises, the conflict degree can be counted directly from the
hardware facts above:

1. **$S$ is a whole number of words.** An odd `chunk` in fp16 makes it a half
   word, so a common divisor is not defined. The remaining method is to compute
   each thread's bank from its byte address and count.
2. **The $32w$ words the 32 threads request are distinct**, that is $S \ge w$.
   Threads landing on one word are broadcast and cost no extra cycle, so
   counting $32w$ words no longer holds; and at $S < w$ the vector ranges of
   adjacent threads overlap, which breaks it the same way.

The warp fetches $32w$ words in total, and shared memory serves at most 32 words
per cycle. The instruction therefore takes at least $w$ cycles. That lower bound
comes from width alone and is independent of the addresses; a conflict is any
cost above it. $St \bmod 32$ takes only multiples of $g = \gcd(S, 32)$, so the
access reaches $32/g$ banks, each hit $g$ times. The shifts
$j = 0, \dots, w-1$ are then added on top. Both $g$ and $w$ are powers of two,
so one divides the other, giving two cases:

| | Where the banks fall | Cycles | Conflict |
| --- | --- | --- | --- |
| $g \le w$ | All 32 banks requested $w$ times each | $w$, exactly the lower bound | None |
| $g > w$ | Only $(32/g) \cdot w$ banks hit, $g$ times each | $g$ | $g / w$-way |

$$\text{conflict degree} \ \ge\ \max\left(1,\ \frac{\gcd(S,\ 32)}{w}\right)$$

For a scalar read ($w = 1$), this is just $\gcd(S, 32)$. When $S$ is coprime
with 32, the 32 threads cover all 32 banks with no conflict. When $S$ is a
multiple of 32, they all land on one bank and serialize 32 ways. fp16 with
`chunk = 64` is the latter case: $S$ is 128 bytes, or 32 words.

The result is an inequality because the last step assumes the hardware can fit
any conflict-free set of words into one cycle. NVIDIA has not published how
lanes are actually grouped for 64-bit and 128-bit accesses.

## Changing the stride with padding

The stride follows from `chunk`, and `chunk` is usually fixed by the algorithm
rather than free to change. One way to remove an $N$-way conflict is to add
`pad` elements to the end of each run, making the stride `chunk + pad`. That
changes the `gcd`, and therefore the conflict degree. For fp16 with
`chunk = 64`, adding just 2 elements changes the stride from 32 words to 33,
which is coprime with 32, and the conflict disappears entirely.

<figure class="bank-conflict" markdown="1">

<svg class="tf-bank" viewBox="0 0 520 354" role="img" aria-label="Where the 32 threads of a warp land across the 32 shared memory banks, for fp16 with 64 elements to a chunk. Unpadded they all land on bank 0, a 32-way conflict; at pad = 2 they cover all 32 banks with no conflict; at pad = 4 two threads share a bank, a 2-way conflict; at pad = 8 four share one, a 4-way conflict.">
<text class="bk-title" x="0" y="43.0">pad = 0</text>
<text class="bk-sub" x="0" y="58.0">stride = 32 words</text>
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
<text class="bk-note" x="150.0" y="66.0">1 of 32 banks used, 32 threads on the busiest — 32-way</text>
<text class="bk-title" x="0" y="117.0">pad = 2</text>
<text class="bk-sub" x="0" y="132.0">stride = 33 words</text>
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
<text class="bk-note" x="150.0" y="140.0">32 of 32 banks used, 1 thread on the busiest — no conflict</text>
<text class="bk-title" x="0" y="191.0">pad = 4</text>
<text class="bk-sub" x="0" y="206.0">stride = 34 words</text>
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
<text class="bk-note" x="150.0" y="214.0">16 of 32 banks used, 2 threads on the busiest — 2-way</text>
<text class="bk-title" x="0" y="265.0">pad = 8</text>
<text class="bk-sub" x="0" y="280.0">stride = 36 words</text>
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
<text class="bk-note" x="150.0" y="288.0">8 of 32 banks used, 4 threads on the busiest — 4-way</text>
<text class="bk-axis" x="155.5" y="22.0" text-anchor="middle">bank 0</text>
<text class="bk-axis" x="496.5" y="22.0" text-anchor="end">31</text>
<text class="bk-scale" x="0" y="337.0">fp16, 64 elements to a chunk, one warp of 32 threads. A number in a cell is</text>
<text class="bk-scale" x="0" y="351.0">how many threads land on that bank; an empty cell means none do.</text>
</svg>

<figcaption>Four views of one kernel at four padding values, showing where the 32 threads of a warp land across the 32 banks. The number in a cell is how many threads land on that bank, and the largest number is the conflict degree. At <code>pad = 0</code>, all 32 threads land on bank 0. Adding 2 elements changes the stride to 33 words, which is coprime with 32, so the 32 threads cover all 32 banks exactly.</figcaption>

</figure>

## How much padding to use

The formula above gives the test, and **both conditions must hold**:

| Access width | $w$ | Conflict requirement | Alignment of a run's start | Fastest pad at fp32, `chunk = 64` |
| --- | --- | --- | --- | --- |
| 32 bit, scalar read | 1 | $\gcd(S, 32) = 1$ | 4 bytes | `pad = 1` ($S = 65$ words) |
| 64 bit, `float2` | 2 | $\gcd(S, 32) \le 2$ | 8 bytes | `pad = 2` ($S = 66$ words) |
| 128 bit, `float4` | 4 | $\gcd(S, 32) \le 4$ | 16 bytes | `pad = 4` ($S = 68$ words) |

The two conditions pull in opposite directions, so neither is sufficient by
itself: **the wider the access, the looser the conflict requirement and the
tighter the alignment.** An odd pad is optimal for a scalar read and the worst
case for a vectorized read, because it shifts the start of each run by 4 bytes.
A 128-bit shared read requires natural 16-byte alignment, and its behavior is
undefined when the alignment is insufficient. If the compiler can see the
insufficient alignment at compile time, it falls back to a narrower instruction;
the $\gcd = 1$ row below shows that case.

The measurements below were taken on an H200 with fp32 and `chunk = 64`. The
consumption loop is repeated 32 times so the shared-memory side becomes the
bottleneck, and time scales proportionally with the repeat count. Each entry is
how many times slower that combination is than the fastest row at the same
width:

| $\gcd(S, 32)$ | 32 bit | 64 bit | 128 bit | Predicted bound (32 / 64 / 128) |
| --- | --- | --- | --- | --- |
| 1 | **1.00** | 1.10 | 2.27 | 1 / 1 / 1 |
| 2 | 1.72 | **1.00** | 1.01 | 2 / 1 / 1 |
| 4 | 3.29 | 1.81 | **1.00** | 4 / 2 / 1 |
| 8 | 6.52 | 3.55 | 4.04 | 8 / 4 / 2 |
| 16 | 12.97 | 7.62 | 7.99 | 16 / 8 / 4 |
| 32 | 25.82 | 13.97 | 15.71 | 32 / 16 / 8 |

The three 1.00 entries on the diagonal are the three recommendations in the
previous table. The bound is tight on the $g \le w$ side. On the $g > w$ side,
it is not tight, and the 128-bit column lands at exactly twice the bound (4.04
versus 2, 7.99 versus 4, 15.71 versus 8). Explaining the cause requires
lower-level evidence, because NVIDIA has not published how lanes are grouped at
those two widths. The 2.27 in the $\gcd = 1$ row comes from alignment:
$S = 65$ words is 260 bytes, not a multiple of 16.

## Padding has to be computed from the chunk {#pad-per-chunk}

Padding by a fixed number of bytes falls back into the worst case for some
values of `chunk`. The alignment requirement restricts the candidates to whole
multiples of 16 bytes. Write the pad as $k$ multiples of 16 bytes
($k = 1, 2, \dots$):

$$S = \frac{\text{chunk} \times E}{4} + 4k \ \text{words}$$

With $k$ fixed, $\gcd(S, 32)$ varies with `chunk`. For fp16 and bf16
($E = 2$), several values of `chunk` give:

| chunk | $S$ ($k = 1$) | $\gcd(S, 32)$ | $S$ ($k = 2$) | $\gcd(S, 32)$ |
| --- | --- | --- | --- | --- |
| 32 | 20 | **4** | 24 | 8 |
| 56 | 32 | 32 | 36 | **4** |
| 64 | 36 | **4** | 40 | 8 |
| 72 | 40 | 8 | 44 | **4** |
| 128 | 68 | **4** | 72 | 8 |

In the `chunk = 56` row, $S$ is exactly 32 words, so the accesses land on the
same bank as with `pad = 0`: padding was added, but the conflict remains. The
fix is to choose whichever of $k = 1$ and $k = 2$ gives the smaller
$\gcd(S, 32)$. Whenever $\text{chunk} \times E$ is a multiple of 16, one of the
two choices is guaranteed to reduce the $\gcd$ to 4. Writing
$q = \text{chunk} \times E / 16$ gives $S = 4(q + k)$ and
$\gcd(S, 32) = 4 \gcd(q + k, 8)$; one of $q + 1$ and $q + 2$ is odd.

The measurements below were taken on an H200 with the SM clock locked at
1830 MHz, bf16, CUPTI device time, L2 flushed before each iteration, median of
200 runs, and image
`ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-afcebed1-dev`. All four rows
use the thread count that makes `chunk` equal 56. Within a row, the two columns
differ only in the pad:

| Input | Threads × chunk | $k = 1$ (pad 8 elements)<br>TB/s | $k = 2$ (pad 16 elements)<br>TB/s |
| --- | --- | --- | --- |
| $2048 \times 3584$ | 64 × 56 | 1.38 | **3.07**{ .win } |
| $2048 \times 7168$ | 128 × 56 | 1.58 | **3.29**{ .win } |
| $1024 \times 14336$ | 256 × 56 | 1.51 | **3.05**{ .win } |
| $512 \times 28672$ | 512 × 56 | 1.32 | **2.33**{ .win } |

Those four widths are the hidden size of Qwen2-7B, the hidden size of
Llama-3-70B, and the FFN intermediate dimensions of Llama-3-8B and
Llama-3-70B. They are not constructed counterexamples.

The shared-memory side of these measurements is not a scalar read. `cuobjdump`
shows 16 `LDS.64` instructions per thread at $S = 36$ words, so $w = 2$ and the
degree is $\gcd(S, 32) / 2$: 16-way and 2-way for the two columns. The degrees
differ by a factor of 8, while bandwidth differs by only 2.2, because the
2-way column is limited by DRAM again. When the start of a run is misaligned by
less than 8 bytes, the compiler falls back to `LDS` ($w = 1$). Width is decided
by the compiler, not by the code that declares the pad, so both columns above
must be measured rather than derived from the $\gcd$ alone.

## A measured sweep

The measurements below read one element per thread at a time ($w = 1$), use
powers of two for `chunk`, and give each thread its own run. Both premises above
therefore hold.

H200, SM clock locked at 1830 MHz. fp16, input $65536 \times 4096$ (512 MB,
larger than the 60 MiB L2). The kernel stages a whole row into shared memory,
with `chunk + pad` elements per thread, computes a serial prefix product over
each run, and writes it back, for 1 GB read and written in total. Parentheses
show the conflict degree predicted by the formula:

| chunk | Threads | pad = 0<br>TB/s | pad = 2<br>TB/s | pad = 4<br>TB/s | pad = 8<br>TB/s | pad = 16<br>TB/s |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 256 | 1.63 (8-way) | 2.99 (1-way) | **3.29**{ .win } (2-way) | 2.58 (4-way) | 0.86 (16-way) |
| 32 | 128 | 0.89 (16-way) | 3.32 (1-way) | **3.35**{ .win } (2-way) | 2.57 (4-way) | 1.54 (8-way) |
| 64 | 64 | 0.46 (32-way) | 3.63 (1-way) | **3.70**{ .win } (2-way) | 2.74 (4-way) | 1.62 (8-way) |
| 128 | 32 | 0.46 (32-way) | 3.25 (1-way) | **3.31**{ .win } (2-way) | 2.76 (4-way) | 1.61 (8-way) |

Across the 20 configurations, 1-way and 2-way stay above 3.0, 4-way drops to
around 2.6, and 8-way and worse fall below 1.6. Bandwidth decreases
monotonically with the predicted degree, so under these conditions the formula
can be used directly to narrow the padding candidates.

## Things to watch when applying this

1. **The formula gives candidates; the final value comes from measurement.** In
   the table above, 1-way and 2-way both stay above 3.0, 4-way drops to around
   2.6, and 8-way and worse fall below 1.6. The formula is useful for narrowing
   the candidates to those that compute to no more than $w$ ways. Among those
   candidates, measure: 2-way edges out 1-way on all four values of `chunk`, but
   only by 0.9% to 10%, which is not a rule worth carrying over.

2. **After changing the pad, sweep `chunk` again.** The best entry in the
   `pad = 0` column is `chunk = 16` (1.63); in the `pad = 4` column, it is
   `chunk = 64` (3.70). The ordering of the first column is set mostly by
   conflict degree: `chunk = 16` hits 8-way, while `chunk = 64` hits 32-way.
   Once the conflicts are removed, all four cases are 2-way, and the optimum
   moves. Thread count and `chunk` move together in this table (their product is
   the row width, 4096), so `chunk` is not the only cause of the move; occupancy
   and loop length change with it. The conclusion is only that the ordering of
   `chunk` changes after the pad changes.

3. **After changing `chunk`, recompute the pad.** Writing the pad as a fixed
   number of bytes assumes $\gcd(S, 32)$ is independent of `chunk`, and
   [the table above](#pad-per-chunk) shows that it is not: at `chunk = 56` and
   $k = 1$, $S$ returns to 32 words. Writing the pad as a function of `chunk`,
   by taking whichever of $k = 1$ and $k = 2$ gives the smaller $\gcd(S, 32)$,
   keeps it consistent with the test on this page.

4. **This page applies only where data passes through shared memory.** By the
   trade-offs in
   [Optimizing Global Memory Access](global-memory-access.md#coalescing), small
   $V$ calls for vectorized blocked. In that pattern, data goes straight into
   registers, never touches shared memory, and has no bank conflicts. This page
   applies once $V$ grows enough that register pressure cuts occupancy and
   staged takes over, and when a whole row has to be shared by every thread in
   the block.

The two listings below declare the shared buffer and differ only in the stride.
The wrong form has a stride that is an exact multiple of 32 words:

```python
sh = T.alloc_shared((threads * chunk,), dtype)          # stride = chunk elements
```

The right form computes the pad from `chunk`, choosing the whole multiple of
16 bytes with the smallest $\gcd(S, 32)$:

```python
import math

def pick_pad(chunk: int, elem_bytes: int) -> int:
    """The pad, in elements, with the smallest gcd(S, 32) among whole multiples of 16 bytes."""
    return min(
        (16 // elem_bytes, 32 // elem_bytes),                    # k = 1, k = 2
        key=lambda pad: math.gcd((chunk + pad) * elem_bytes // 4, 32),
    )

pad = pick_pad(chunk, elem_bytes)                                # still measure between candidates
sh = T.alloc_shared((threads * (chunk + pad),), dtype)           # stride = chunk + pad
```
