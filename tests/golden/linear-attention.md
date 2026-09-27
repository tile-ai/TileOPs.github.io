# Linear Attention

**1 ops, 3 workloads.**

One table per op, one row per workload. `Ratio` is the fastest other implementation's device time divided by ours, so <span class="perf-ahead">green</span> is faster than it, <span class="perf-par">plain</span> is level with it, <span class="perf-behind">red</span> is slower. Times are in ms. [How these numbers are taken](reading.md).

## [DeltaDecodeFwd](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+DeltaDecodeFwdOp&type=code)

<div class="wl-key">
<div class="wl-group"><p class="wl-shared"><span class="wl-cell wl-tensor"><span class="wl-k">q, k</span>: [B, H, DK]</span><span class="wl-cell wl-tensor"><span class="wl-k">v</span>: [B, H, DV]</span><span class="wl-cell wl-tensor"><span class="wl-k">state</span>: [B, H, DK, DV]</span></p><ul class="wl-rows"><li><code class="wl-id">decode-<wbr>b1-<wbr>h8</code><span class="wl-delta"><span class="wl-flow"><span class="wl-part"><span class="wl-cell wl-tensor"><span class="wl-k">q, k, v</span>: [1, 8, 128]</span><span class="wl-cell wl-tensor"><span class="wl-k">state</span>: [1, 8, 128, 128]</span></span><span class="wl-part"><span class="wl-cell wl-scalar"><span class="wl-k">dtype</span>=<span class="wl-v">bf16, f16</span></span></span></span></span></li><li><code class="wl-id">decode-<wbr>b8-<wbr>h8</code><span class="wl-delta"><span class="wl-flow"><span class="wl-part"><span class="wl-cell wl-tensor"><span class="wl-k">q, k, v</span>: [8, 8, 128]</span><span class="wl-cell wl-tensor"><span class="wl-k">state</span>: [8, 8, 128, 128]</span></span><span class="wl-part"><span class="wl-cell wl-scalar"><span class="wl-k">dtype</span>=<span class="wl-v">bf16</span></span></span></span></span></li></ul></div>
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
<tr><td class="wl-name" rowspan="2"><code>decode-<wbr>b1-<wbr>h8</code></td><td class="colsep">bf16</td><td><span class="perf-ahead">4.58×</span></td><td>0.0031</td><td><code>fla</code><br><span class="alt-slow"><code>torch-ref</code></span></td><td>0.0142<br><span class="alt-slow">0.0353</span></td><td>0.508</td><td>·</td><td>·</td></tr>
<tr><td class="colsep">f16</td><td><span class="perf-ahead">4.30×</span></td><td>0.0033</td><td><code>fla</code><br><span class="alt-slow"><code>torch-ref</code></span></td><td>0.0142<br><span class="alt-slow">0.0353</span></td><td>0.508</td><td>·</td><td>·</td></tr>
<tr><td class="wl-name"><code>decode-<wbr>b8-<wbr>h8</code></td><td class="colsep">bf16</td><td><span class="perf-behind">0.50×</span></td><td>0.0200</td><td><code>fla</code></td><td>0.0100</td><td>0.629</td><td>·</td><td>·</td></tr>
</tbody>
</table>
</div>

