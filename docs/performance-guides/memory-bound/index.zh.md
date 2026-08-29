# 优化访存受限的 kernel

## 什么是访存受限

在 GPU 上，一个 kernel 跑多快，取决于算力与带宽哪一个先成为瓶颈。TileOPs 用 [macro benchmark](https://github.com/tile-ai/TileOPs/tree/main/benchmarks/hardware) 测算出一个**[校准系数](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/perf/profiles/h200.yaml)**：硬件 spec 给出的理论峰值乘以校准系数，得到实际可达的有效值，以此作为性能优化的指导标准。我们在 H200 上实测出 [fp32 FMA 的算力为 **57.27** TFLOP/s，访存带宽为 **4.07** TB/s](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/perf/profiles/h200.yaml)。roofline 的**拐点**（ridge point）是带宽斜线与算力上限这两段的交点，两者相除给出它的横坐标，也就是拐点处的算术强度 **14.07 flop/byte** —— 算力与带宽同时用满时，每搬运一个字节对应的浮点运算次数：

<figure class="roofline" markdown="1">

<svg class="tf-roofline" viewBox="0 0 520 306" role="img" aria-label="H200 的 roofline：带宽上限 4.07 TB/s 的斜线在每字节 14 次浮点运算处与 57.27 TFLOP/s 的算力上限相交；silu 位于斜线左端，可达算力为算力上限的 9%。">
<path class="tf-rl-region" d="M 52.0 250.0 L 52.0 217.9 L 357.4 67.1 L 357.4 250.0 Z"/>
<line class="tf-rl-grid" x1="115.4" y1="52" x2="115.4" y2="250"/>
<text class="tf-rl-tick" x="115.4" y="270" text-anchor="middle">1</text>
<line class="tf-rl-grid" x1="178.9" y1="52" x2="178.9" y2="250"/>
<text class="tf-rl-tick" x="178.9" y="270" text-anchor="middle">2</text>
<line class="tf-rl-grid" x1="242.3" y1="52" x2="242.3" y2="250"/>
<text class="tf-rl-tick" x="242.3" y="270" text-anchor="middle">4</text>
<line class="tf-rl-grid" x1="305.7" y1="52" x2="305.7" y2="250"/>
<text class="tf-rl-tick" x="305.7" y="270" text-anchor="middle">8</text>
<line class="tf-rl-grid" x1="369.1" y1="52" x2="369.1" y2="250"/>
<text class="tf-rl-tick" x="369.1" y="270" text-anchor="middle">16</text>
<line class="tf-rl-grid" x1="432.6" y1="52" x2="432.6" y2="250"/>
<text class="tf-rl-tick" x="432.6" y="270" text-anchor="middle">32</text>
<line class="tf-rl-grid" x1="496.0" y1="52" x2="496.0" y2="250"/>
<text class="tf-rl-tick" x="496.0" y="270" text-anchor="middle">64</text>
<line class="tf-rl-grid" x1="52" y1="250.0" x2="496" y2="250.0"/>
<text class="tf-rl-tick" x="44" y="255.0" text-anchor="end">1</text>
<line class="tf-rl-grid" x1="52" y1="218.7" x2="496" y2="218.7"/>
<text class="tf-rl-tick" x="44" y="223.7" text-anchor="end">2</text>
<line class="tf-rl-grid" x1="52" y1="187.4" x2="496" y2="187.4"/>
<text class="tf-rl-tick" x="44" y="192.4" text-anchor="end">4</text>
<line class="tf-rl-grid" x1="52" y1="156.0" x2="496" y2="156.0"/>
<text class="tf-rl-tick" x="44" y="161.0" text-anchor="end">8</text>
<line class="tf-rl-grid" x1="52" y1="124.7" x2="496" y2="124.7"/>
<text class="tf-rl-tick" x="44" y="129.7" text-anchor="end">16</text>
<line class="tf-rl-grid" x1="52" y1="93.4" x2="496" y2="93.4"/>
<text class="tf-rl-tick" x="44" y="98.4" text-anchor="end">32</text>
<line class="tf-rl-grid" x1="52" y1="62.1" x2="496" y2="62.1"/>
<text class="tf-rl-tick" x="44" y="67.1" text-anchor="end">64</text>
<line class="tf-rl-axis" x1="52" y1="250" x2="496" y2="250"/>
<line class="tf-rl-axis" x1="52" y1="52" x2="52" y2="250"/>
<text class="tf-rl-axis-title" x="4" y="34">可达算力 TFLOP/s（对数轴）</text>
<text class="tf-rl-axis-title" x="496" y="294" text-anchor="end">算术强度：每字节的浮点运算数（对数轴）</text>
<polyline class="tf-rl-roof" points="52.0,217.9 357.4,67.1 496.0,67.1"/>
<line class="tf-rl-drop" x1="357.4" y1="67.1" x2="357.4" y2="250"/>
<circle class="tf-rl-ridge" cx="357.4" cy="67.1" r="6"/>
<text class="tf-rl-label tf-rl-label--ridge" x="370.4" y="89.1">拐点</text>
<text class="tf-rl-sub" x="370.4" y="107.1">每字节 14 次</text>
<circle class="tf-rl-point" cx="135.8" cy="176.5" r="5.5"/>
<text class="tf-rl-label tf-rl-label--point" x="148.8" y="196.5">silu (fp16)</text>
<text class="tf-rl-sub" x="148.8" y="214.5">5 次 / 4 字节，上限 5.1</text>
<text class="tf-rl-roof-label" x="152.5" y="153.3" transform="rotate(-30 152.5 153.3)">带宽上限 4.07 TB/s</text>
<text class="tf-rl-roof-label" x="492.0" y="54.1" text-anchor="end">算力上限 57.3 TFLOP/s</text>
</svg>

<figcaption>图上这条折线就是 roofline，任何 kernel 的性能点都落在它以下。拐点以左，上限是「算术强度 × 带宽」，可达算力随算术强度线性上升；拐点以右，上限就是算力峰值，不再随算术强度变化。<code>silu</code> 每搬运 4 个字节做 5 次运算，算术强度 1.25 flop/byte，只有拐点的 1/11，所以即便带宽完全用满，也只能达到算力上限的 9%。</figcaption>

</figure>

1. [优化 global memory 访问](global-memory-access.md)
2. [优化 shared memory 访问](shared-memory-access.md)
