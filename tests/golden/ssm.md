# SSM

**1 ops, 2 workloads.**

One table per op, one row per workload. `Ratio` is the fastest other implementation's device time divided by ours, so <span class="perf-ahead">green</span> is faster than it, <span class="perf-par">plain</span> is level with it, <span class="perf-behind">red</span> is slower. Times are in ms. [How these numbers are taken](reading.md).

## [ChunkScanFwd](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+ChunkScanFwdOp&type=code) <small>❌</small>

<div class="wl-key">
<div class="wl-group"><p class="wl-shared"><span class="wl-cell wl-scalar"><span class="wl-k">dtype</span>=<span class="wl-v">bf16</span></span><span class="wl-cell wl-scalar"><span class="wl-k">num_chunks</span>=<span class="wl-v">4</span></span><span class="wl-cell wl-scalar"><span class="wl-k">chunk_len</span>=<span class="wl-v">64</span></span><span class="wl-cell wl-scalar"><span class="wl-k">N</span>=<span class="wl-v">128</span></span></p><ul class="wl-rows"><li><b>W1</b><span class="wl-delta"><span class="wl-cell wl-scalar"><span class="wl-k">batch</span>=<span class="wl-v">2</span></span></span><code class="wl-id">scan-b2</code></li><li><b>W2</b><span class="wl-delta"><span class="wl-cell wl-scalar"><span class="wl-k">batch</span>=<span class="wl-v">4</span></span><span class="wl-cell wl-scalar"><span class="wl-dim"><span class="wl-k">is_causal</span>=<span class="wl-v">false</span></span></span></span><code class="wl-id">scan-b4</code></li></ul></div>
</div>

<div class="datatable">
<table>
<thead>
<tr>
<th rowspan="2" class="colsep">Workload</th>
<th>Ratio</th>
<th>Device time</th>
<th colspan="2">Alternatives</th>
<th>Throughput</th>
<th>SOL</th>
<th>Bound</th>
</tr>
<tr>
<th class="subhead">alt / ours</th>
<th class="subhead">ms</th>
<th class="subhead">name</th>
<th class="subhead">ms</th>
<th class="subhead">TFLOP/s</th>
<th class="subhead">of ceiling</th>
<th class="subhead">by</th>
</tr>
</thead>
<tbody>
<tr><td class="colsep"><b>W1</b></td><td><span class="perf-unrated">4.00×</span></td><td>0.0500</td><td><code>torch-ref</code></td><td>0.2000</td><td>·</td><td>·</td><td>·</td></tr>
<tr><td class="colsep"><b>W2</b></td><td><span class="perf-ahead">1.60×</span></td><td>0.1000</td><td><code>torch</code></td><td>0.1000</td><td>·</td><td>·</td><td>·</td></tr>
</tbody>
</table>
</div>

