# Elementwise

**3 ops, 4 workloads.**

One table per op, one row per workload. `Ratio` is the fastest other implementation's device time divided by ours, so <span class="perf-ahead">green</span> is faster than it, <span class="perf-par">plain</span> is level with it, <span class="perf-behind">red</span> is slower. Times are in ms. [How these numbers are taken](reading.md).

## [MysteryFwd](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+MysteryFwdOp&type=code) <small>⏭️</small>

<div class="wl-key">
<div class="wl-group"><ul class="wl-rows"><li><code class="wl-id">undeclared-<wbr>op-<wbr>case</code><span class="wl-delta"><span class="wl-flow"><span class="wl-part"><span class="wl-cell wl-scalar"><span class="wl-k">dtype</span>=<span class="wl-v">f16</span></span></span></span></span></li></ul></div>
</div>

<div class="datatable">
<table>
<thead>
<tr>
<th rowspan="2">Workload</th>
<th rowspan="2" class="colsep">dtype</th>
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
<tr><td class="wl-name"><code>undeclared-<wbr>op-<wbr>case</code></td><td class="colsep">f16</td><td><span class="perf-ahead">2.00×</span></td><td>0.004</td><td><code>triton</code><br><span class="alt-slow"><code>brand-new-lib</code></span></td><td>0.008<br><span class="alt-slow">0.0120</span></td><td>·</td><td>·</td><td>·</td></tr>
</tbody>
</table>
</div>

## [SquareFwd](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+SquareFwdOp&type=code)

<div class="wl-key">
<div class="wl-group"><ul class="wl-rows"><li><code class="wl-id">oblong</code><span class="wl-delta"><span class="wl-flow"><span class="wl-part"><span class="wl-cell wl-tensor"><span class="wl-k">a</span>: [64, 32]</span></span><span class="wl-part"><span class="wl-cell wl-scalar"><span class="wl-k">dtype</span>=<span class="wl-v">f16</span></span></span></span></span></li></ul></div>
</div>

<div class="datatable">
<table>
<thead>
<tr>
<th rowspan="2">Workload</th>
<th rowspan="2" class="colsep">dtype</th>
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
<tr><td class="wl-name"><code>oblong</code></td><td class="colsep">f16</td><td><span class="perf-ahead">1.50×</span></td><td>0.002</td><td><code>torch</code></td><td>0.003</td><td>·</td><td>·</td><td>·</td></tr>
</tbody>
</table>
</div>

## [TemplatedFwd](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+TemplatedFwdOp&type=code)

<div class="wl-key">
<div class="wl-group"><p class="wl-shared"><span class="wl-cell wl-tensor"><span class="wl-k">x</span>: [rows, cols]</span><span class="wl-cell wl-tensor"><span class="wl-k">mask</span>: [rows], <span class="wl-dt">bool</span></span></p><p class="wl-shared"><span class="wl-cell wl-scalar"><span class="wl-k">dtype</span>=<span class="wl-v">f16</span></span></p><ul class="wl-rows"><li><code class="wl-id">templated-<wbr>64x256</code><span class="wl-delta"><span class="wl-flow"><span class="wl-part"><span class="wl-cell wl-tensor"><span class="wl-k">x</span>: [64, 256]</span><span class="wl-cell wl-tensor"><span class="wl-k">mask</span>: [64], <span class="wl-dt">bool</span></span></span></span></span></li><li><code class="wl-id">templated-<wbr>128x256</code><span class="wl-delta"><span class="wl-flow"><span class="wl-part"><span class="wl-cell wl-tensor"><span class="wl-k">x</span>: [128, 256]</span><span class="wl-cell wl-tensor"><span class="wl-k">mask</span>: [128], <span class="wl-dt">bool</span></span></span></span></span></li></ul></div>
</div>

<div class="datatable">
<table>
<thead>
<tr>
<th rowspan="2">Workload</th>
<th rowspan="2" class="colsep">dtype</th>
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
<tr><td class="wl-name"><code>templated-<wbr>64x256</code></td><td class="colsep">f16</td><td><span class="perf-ahead">1.50×</span></td><td>0.006</td><td><code>torch</code></td><td>0.009</td><td>·</td><td>·</td><td>·</td></tr>
<tr><td class="wl-name"><code>templated-<wbr>128x256</code></td><td class="colsep">f16</td><td><span class="perf-ahead">1.43×</span></td><td>0.007</td><td><code>torch</code></td><td>0.0100</td><td>·</td><td>·</td><td>·</td></tr>
</tbody>
</table>
</div>

