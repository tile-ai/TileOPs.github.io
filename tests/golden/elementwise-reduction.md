# Elementwise & Reduction

**3 ops, 4 workloads.**

One table per op, one row per workload. `Ratio` is the fastest other implementation's device time divided by ours, so <span class="perf-ahead">green</span> is faster than it, <span class="perf-par">plain</span> is level with it, <span class="perf-behind">red</span> is slower. Times are in ms. [How these numbers are taken](reading.md).

## Elementwise

### [MysteryFwd](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+MysteryFwdOp&type=code) <small>⏭️</small>

<div class="wl-key">
<div class="wl-group"><ul class="wl-rows"><li><b>W1</b><span class="wl-delta"></span><code class="wl-id">undeclared-op-case-float16</code></li></ul></div>
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
<tr><td class="colsep"><b>W1</b></td><td><span class="perf-ahead">2.00×</span></td><td>0.004</td><td><code>triton</code><br><span class="alt-slow"><code>brand-new-lib</code></span></td><td>0.008<br><span class="alt-slow">0.0120</span></td><td>·</td><td>·</td><td>·</td></tr>
</tbody>
</table>
</div>

### [SquareFwd](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+SquareFwdOp&type=code)

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

### [TemplatedFwd](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+TemplatedFwdOp&type=code)

<div class="wl-key">
<div class="wl-group"><p class="wl-shared"><span class="wl-cell wl-tensor"><span class="wl-k">x</span>: [rows, cols]</span><span class="wl-cell wl-tensor"><span class="wl-k">mask</span>: [rows], <span class="wl-dt">bool</span></span></p><p class="wl-shared"><span class="wl-cell wl-scalar"><span class="wl-k">dtype</span>=<span class="wl-v">f16</span></span><span class="wl-cell wl-scalar"><span class="wl-k">cols</span>=<span class="wl-v">256</span></span></p><ul class="wl-rows"><li><b>W1</b><span class="wl-delta"><span class="wl-cell wl-scalar"><span class="wl-k">rows</span>=<span class="wl-v">128</span></span></span><code class="wl-id">templated-128x256</code></li><li><b>W2</b><span class="wl-delta"><span class="wl-cell wl-scalar"><span class="wl-k">rows</span>=<span class="wl-v">64</span></span></span><code class="wl-id">templated-64x256</code></li></ul></div>
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
<tr><td class="colsep"><b>W1</b></td><td><span class="perf-ahead">1.43×</span></td><td>0.007</td><td><code>torch</code></td><td>0.0100</td><td>·</td><td>·</td><td>·</td></tr>
<tr><td class="colsep"><b>W2</b></td><td><span class="perf-ahead">1.50×</span></td><td>0.006</td><td><code>torch</code></td><td>0.009</td><td>·</td><td>·</td><td>·</td></tr>
</tbody>
</table>
</div>

