# Optimizing Shared Memory Access

Routing data through shared memory adds one more place where the access pattern
matters: writing it in from global memory and reading it out into registers both
access shared memory. This page covers bank conflicts on those two steps — what
determines them, how padding removes them, and what the padding should be
computed from.

Every measurement on this page was taken on an H200 with the SM clock locked at
1830 MHz, an input larger than the 60 MiB L2, and enough blocks to fill the
whole card. Outside those conditions the conclusions can reverse; the tests are
in [Optimizing Global Memory Access](global-memory-access.md#regime).

## The bank structure of shared memory {#bank-conflict}

Shared memory is built from 32 banks, each 4 bytes wide. Dividing its address
space into 4-byte **words** (everything below counts in words), which bank an
address falls on is `(byte address / 4) mod 32`. A bank serves one word per
cycle, so when several threads access shared memory concurrently and their
accesses land on different banks, they all complete in the same cycle. When
several threads land on the same bank, there are three cases:

1. **Different words** — the hardware splits the request into several
   conflict-free requests served one after another, and the number of splits is
   the **conflict degree**.
2. **Reading the same word** — any two threads landing inside one word (even on
   different bytes of it) get that word broadcast to all of them, with no
   conflict. Several broadcasts on different banks are further merged into one
   multicast.
3. **Writing the same address** — one write takes effect, and which one is
   undefined.

An access pattern on shared memory should be conflict-free wherever it can be.

## What determines the conflict degree

Consider a case general enough to reason from. The one-dimensional array `sh`
below lives in shared memory, and each thread reads `chunk` contiguous elements
of it:

```python
sh = T.alloc_shared((threads * chunk,), dtype)   # one dimension, threads * chunk elements

for c in T.serial(chunk):
    acc[0] = acc[0] * sh[tx * chunk + c]         # thread tx reads its own run
```

The 32 threads of a warp execute this loop in lockstep, and within one iteration
`c` takes the same value for all of them while `tx` runs 0 to 31: the 32 threads
access shared memory at a fixed spacing. At `chunk = 64`, for one iteration they
read elements `c`, `64 + c`, `128 + c`, …, `1984 + c`. Adjacent threads are
`chunk` elements apart, and that difference is the **stride** of the access,
written $S$; in words it is $S = \text{chunk} \times E / 4$, where $E$ is the
size of an element in bytes. The stride has nothing to do with how many
dimensions the array is declared with or how the index is written.

The other variable is the width of the memory instruction. A vectorized access
reads $w$ contiguous words at a time, where $w$ is 1, 2, or 4 — 32 bit, the
8 bytes of a `float2`, the 16 bytes of a `float4`. Thread $t$ then reads words
$St$ through $St + w - 1$.

Given two premises, the conflict degree can be counted directly from the
hardware facts above:

1. **$S$ is a whole number of words.** An odd `chunk` in fp16 makes it half a
   word, which leaves no common divisor to speak of; the only way left is to
   work out each thread's bank from its byte address and count.
2. **The $32w$ words the 32 threads request are distinct**, that is $S \ge w$.
   Threads landing on one word are broadcast and cost no extra cycle, so
   counting $32w$ words no longer holds; and at $S < w$ the vector ranges of
   adjacent threads overlap, which breaks it the same way.

The warp fetches $32w$ words in all and shared memory serves at most 32 per
cycle, so the instruction takes at least $w$ cycles — a lower bound set by the
width alone, independent of the addresses, and a conflict is what exceeds it.
$St \bmod 32$ only takes multiples of $g = \gcd(S, 32)$, which is $32/g$ banks,
each hit $g$ times; on top of that come the shifts $j = 0, \dots, w-1$. Both $g$
and $w$ are powers of two, so one divides the other, and there are two cases:

| | Where the banks fall | Cycles | Conflict |
| --- | --- | --- | --- |
| $g \le w$ | All 32 banks requested $w$ times each | $w$, exactly the lower bound | None |
| $g > w$ | Only $(32/g) \cdot w$ banks hit, $g$ times each | $g$ | $g / w$-way |

$$\text{conflict degree} \ \ge\ \max\left(1,\ \frac{\gcd(S,\ 32)}{w}\right)$$

For a scalar read ($w = 1$) this is just $\gcd(S, 32)$: when $S$ is coprime with
32 the 32 threads cover all 32 banks with no conflict, and when $S$ is a
multiple of 32 they all pile onto one bank, serialized 32 ways — fp16 with
`chunk = 64` is the latter, where $S$ is 128 bytes, or 32 words.

It is an inequality because the last step assumes the hardware can fit any
conflict-free set of words into one cycle, and NVIDIA has not published how
lanes are actually grouped for 64-bit and 128-bit accesses.

## Changing the stride with padding

The stride follows from `chunk`, and `chunk` is usually fixed by the algorithm
and not free to change. One way out of an $N$-way conflict is to add `pad`
elements to the end of each run, making the stride `chunk + pad` — the `gcd`
changes with it, and so does the conflict degree. For fp16 with `chunk = 64`,
adding just 2 elements takes the stride from 32 words to 33, coprime with 32,
and the conflict disappears entirely.

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

<figcaption>Four views of one kernel at four padding values, showing where the 32 threads of a warp land across the 32 banks. The number in a cell is how many threads land on that bank, and the largest of those numbers is the conflict degree. At <code>pad = 0</code> all 32 threads pile onto bank 0; adding 2 elements takes the stride to 33 words, coprime with 32, and the 32 threads cover all 32 banks exactly.</figcaption>

</figure>

## How much padding to use

The formula above gives the test, and **both rows have to hold**:

| Access width | $w$ | Conflict requirement | Alignment of a run's start | Fastest pad at fp32, `chunk = 64` |
| --- | --- | --- | --- | --- |
| 32 bit, scalar read | 1 | $\gcd(S, 32) = 1$ | 4 bytes | `pad = 1` ($S = 65$ words) |
| 64 bit, `float2` | 2 | $\gcd(S, 32) \le 2$ | 8 bytes | `pad = 2` ($S = 66$ words) |
| 128 bit, `float4` | 4 | $\gcd(S, 32) \le 4$ | 16 bytes | `pad = 4` ($S = 68$ words) |

The two pull in opposite directions, so neither can be read alone: **the wider
the access, the looser the conflict requirement and the tighter the alignment.**
An odd pad is optimal for a scalar read and the worst case for a vectorized one,
because it puts the start of each run 4 bytes out of step. A 128-bit shared read
requires natural 16-byte alignment, and its behaviour is undefined when the
alignment falls short; where the compiler can see that at compile time it falls
back to a narrower instruction, which is what the $\gcd = 1$ row below shows.

Measured on an H200 (fp32, `chunk = 64`, the consumption loop repeated 32 times
so the shared memory side is the limit, with time scaling proportionally),
giving how many times slower each combination is than the fastest row at the
same width:

| $\gcd(S, 32)$ | 32 bit | 64 bit | 128 bit | Predicted bound (32 / 64 / 128) |
| --- | --- | --- | --- | --- |
| 1 | **1.00** | 1.10 | 2.27 | 1 / 1 / 1 |
| 2 | 1.72 | **1.00** | 1.01 | 2 / 1 / 1 |
| 4 | 3.29 | 1.81 | **1.00** | 4 / 2 / 1 |
| 8 | 6.52 | 3.55 | 4.04 | 8 / 4 / 2 |
| 16 | 12.97 | 7.62 | 7.99 | 16 / 8 / 4 |
| 32 | 25.82 | 13.97 | 15.71 | 32 / 16 / 8 |

The three 1.00 entries on the diagonal are the three recommendations in the
previous table. The bound is tight on the $g \le w$ side; on the $g > w$ side it
is not, and the 128-bit column comes out at exactly twice it (4.04 against 2,
7.99 against 4, 15.71 against 8). Settling the cause needs lower-level evidence
than this, since NVIDIA has not published how lanes are grouped at those two
widths. The 2.27 in the $\gcd = 1$ row is alignment: $S = 65$ words is 260
bytes, not a multiple of 16.

## Padding has to be computed from the chunk {#pad-per-chunk}

A padding fixed in bytes falls back into the worst case at some values of
`chunk`. The alignment requirement restricts the candidates to whole multiples
of 16 bytes, so writing the pad as $k$ lots of 16 bytes ($k = 1, 2, \dots$),

$$S = \frac{\text{chunk} \times E}{4} + 4k \ \text{words}$$

With $k$ fixed, $\gcd(S, 32)$ varies with `chunk`. A few values of `chunk` for
fp16 and bf16 ($E = 2$):

| chunk | $S$ ($k = 1$) | $\gcd(S, 32)$ | $S$ ($k = 2$) | $\gcd(S, 32)$ |
| --- | --- | --- | --- | --- |
| 32 | 20 | **4** | 24 | 8 |
| 56 | 32 | 32 | 36 | **4** |
| 64 | 36 | **4** | 40 | 8 |
| 72 | 40 | 8 | 44 | **4** |
| 128 | 68 | **4** | 72 | 8 |

In the `chunk = 56` row $S$ comes out at exactly 32 words, landing on the same
bank as `pad = 0` — the padding was added and the conflict stayed. The way round
it is to take whichever of $k = 1$ and $k = 2$ gives the smaller $\gcd(S, 32)$.
Whenever $\text{chunk} \times E$ is a multiple of 16, one of the two is
guaranteed to bring the $\gcd$ down to 4: writing
$q = \text{chunk} \times E / 16$ gives $S = 4(q + k)$ and
$\gcd(S, 32) = 4 \gcd(q + k, 8)$, and one of $q + 1$ and $q + 2$ is odd.

Measured on an H200 (SM clock locked at 1830 MHz, bf16, CUPTI device time, L2
flushed before each iteration, median of 200, image
`ghcr.io/tile-ai/tileops-runner:cu132-torch2.13-tl-afcebed1-dev`). All four rows
pick the thread count that makes `chunk` equal 56, and the two columns of a row
differ only in the pad:

| Input | Threads × chunk | $k = 1$ (pad 8 elements)<br>TB/s | $k = 2$ (pad 16 elements)<br>TB/s |
| --- | --- | --- | --- |
| $2048 \times 3584$ | 64 × 56 | 1.38 | **3.07**{ .win } |
| $2048 \times 7168$ | 128 × 56 | 1.58 | **3.29**{ .win } |
| $1024 \times 14336$ | 256 × 56 | 1.51 | **3.05**{ .win } |
| $512 \times 28672$ | 512 × 56 | 1.32 | **2.33**{ .win } |

Those four widths are the hidden size of Qwen2-7B, the hidden size of
Llama-3-70B, and the FFN intermediate dimensions of Llama-3-8B and Llama-3-70B —
not constructed counterexamples.

The shared side of these measurements is not a scalar read: `cuobjdump` shows 16
`LDS.64` per thread at $S = 36$ words, so $w = 2$ and the degree is
$\gcd(S, 32) / 2$ — 16-way and 2-way for the two columns. The degrees differ by
a factor of 8 while the bandwidth differs by 2.2, because in the 2-way column
the limit has moved back to DRAM. When a run starts less than 8 bytes out of
step it falls back to `LDS` ($w = 1$) — the width is the compiler's decision,
not that of whoever declares the pad, so both columns above have to be measured
rather than derived from the $\gcd$ alone.

## A measured sweep

The measurements below read one element per thread at a time ($w = 1$), take
`chunk` at powers of two, and give each thread its own run — so both premises
above hold.

H200, SM clock locked at 1830 MHz. fp16, input $65536 \times 4096$ (512 MB,
larger than the 60 MiB L2). The kernel stages a whole row into shared memory,
`chunk + pad` elements per thread, computes a serial prefix product over each
run and writes it back, 1 GB read and written in total. In parentheses is the
conflict degree the formula predicts:

| chunk | Threads | pad = 0<br>TB/s | pad = 2<br>TB/s | pad = 4<br>TB/s | pad = 8<br>TB/s | pad = 16<br>TB/s |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 256 | 1.63 (8-way) | 2.99 (1-way) | **3.29**{ .win } (2-way) | 2.58 (4-way) | 0.86 (16-way) |
| 32 | 128 | 0.89 (16-way) | 3.32 (1-way) | **3.35**{ .win } (2-way) | 2.57 (4-way) | 1.54 (8-way) |
| 64 | 64 | 0.46 (32-way) | 3.63 (1-way) | **3.70**{ .win } (2-way) | 2.74 (4-way) | 1.62 (8-way) |
| 128 | 32 | 0.46 (32-way) | 3.25 (1-way) | **3.31**{ .win } (2-way) | 2.76 (4-way) | 1.61 (8-way) |

Across the 20 configurations, 1-way and 2-way stay above 3.0, 4-way drops to
around 2.6, and 8-way and worse fall below 1.6. Bandwidth falls monotonically
with the predicted degree, so under these conditions the formula can be used
directly to narrow the padding candidates.

## Things to watch when applying this

1. **The formula gives candidates; the final value comes from measurement.** In
   the table above 1-way and 2-way both stay above 3.0, 4-way drops to around
   2.6, and 8-way and worse fall below 1.6, so what the formula is good for is
   narrowing the candidates to those that compute to no more than $w$ ways.
   Between those, measure: 2-way edges out 1-way on all four values of `chunk`,
   but by anywhere from 0.9% to 10%, which is not a rule worth carrying over.

2. **After changing the pad, sweep `chunk` again.** The best entry in the
   `pad = 0` column is `chunk = 16` (1.63); in the `pad = 4` column it is
   `chunk = 64` (3.70). The ordering of the first column is set mostly by
   conflict degree — `chunk = 16` hits 8-way, `chunk = 64` hits 32-way. Once the
   conflicts are gone all four are 2-way and the optimum has moved. Thread count
   and `chunk` move together in this table (their product is the row width,
   4096), so `chunk` is not the only cause of the move — occupancy and loop
   length change with it. All that can be established is that the ordering of
   `chunk` changes once the pad does.

3. **After changing `chunk`, recompute the pad.** Writing the pad as a fixed
   number of bytes assumes $\gcd(S, 32)$ is independent of `chunk`, and
   [the table above](#pad-per-chunk) shows it is not: at `chunk = 56` and
   $k = 1$, $S$ returns to 32 words. Writing the pad as a function of `chunk` —
   taking whichever of $k = 1$ and $k = 2$ gives the smaller $\gcd(S, 32)$ — is
   what keeps it consistent with the test on this page.

4. **This page applies only where data passes through shared memory.** By the
   trade-offs in
   [Optimizing Global Memory Access](global-memory-access.md#coalescing), a
   small $V$ calls for vectorized blocked, where data goes straight into
   registers, never touches shared memory, and has no bank conflicts to speak
   of. This page applies once $V$ grows enough that register pressure cuts
   occupancy and staged takes over, and where a whole row has to be shared by
   every thread in the block.

The two listings below declare the shared buffer, and differ only in the stride.
The wrong form, where the stride is an exact multiple of 32 words:

```python
sh = T.alloc_shared((threads * chunk,), dtype)          # stride = chunk elements
```

The right form, where the pad is computed from `chunk`, taking the whole
multiple of 16 bytes with the smallest $\gcd(S, 32)$:

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
