# Elementwise & Reduction

**2 ops, 2 workloads.**

One table per op, one row per workload. `Ratio` is the fastest other implementation's device time divided by ours, so <span class="perf-ahead">green</span> is faster than it, <span class="perf-par">plain</span> is level with it, <span class="perf-behind">red</span> is slower. Times are in ms. [How these numbers are taken](reading.md).

## Elementwise

### [MysteryFwd](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+MysteryFwdOp&type=code) <small>(1 workloads · ⏭️)</small>

<div class="wl-key">
<li><b>W1</b><span class="wl-delta"></span><code class="wl-id">undeclared-op-case-float16</code></li>
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
<tr><td class="colsep"><b>W1</b></td><td><span class="perf-ahead">2.00×</span></td><td>0.004</td><td><code>triton</code></td><td>0.008</td><td>·</td><td>·</td><td>·</td></tr>
</tbody>
</table>
</div>

### [SquareFwd](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+SquareFwdOp&type=code) <small>(1 workloads)</small>

<div class="wl-key">
<div class="wl-group"><p class="wl-shared"><span class="wl-cell wl-scalar"><span class="wl-k">dtype</span>=<span class="wl-v">f16</span></span></p><ul class="wl-rows"><li><b>W1</b><span class="wl-delta"><span class="wl-cell wl-tensor"><span class="wl-k">a</span>: [64, 32]</span></span><code class="wl-id">oblong</code></li></ul></div>
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
<tr><td class="colsep"><b>W1</b></td><td><span class="perf-ahead">1.50×</span></td><td>0.002</td><td><code>torch</code></td><td>0.003</td><td>·</td><td>·</td><td>·</td></tr>
</tbody>
</table>
</div>

