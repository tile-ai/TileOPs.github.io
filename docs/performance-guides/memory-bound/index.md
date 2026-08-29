# Tuning Memory-Bound Kernels

## What memory-bound means

On a GPU, kernel speed is determined by whether compute or bandwidth becomes
the bottleneck first. TileOPs measures a
**[calibration factor](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/perf/profiles/h200.yaml)**
with a [macro benchmark](https://github.com/tile-ai/TileOPs/tree/main/benchmarks/hardware):
the hardware peak from the specification, multiplied by that factor, gives the
effective peak a kernel can actually reach. That effective peak is the reference
point for tuning. On an H200, the measured values are
[**57.27** TFLOP/s for fp32 FMA and **4.07** TB/s of memory bandwidth](https://github.com/tile-ai/TileOPs/blob/main/src/tileops/perf/profiles/h200.yaml).
The **ridge point** of the roofline is where the bandwidth slope meets the
compute ceiling. Dividing compute by bandwidth places it at an arithmetic
intensity of **14.07 flop/byte**: the number of floating-point operations per
byte moved when compute and bandwidth are saturated at the same time:

<figure class="roofline" markdown="1">

<svg class="tf-roofline" viewBox="0 0 520 306" role="img" aria-label="Roofline for an H200: the 4.07 TB/s bandwidth slope meets the 57.27 TFLOP/s compute ceiling at 14 flop per byte; silu sits far down the slope, reaching 9% of the compute ceiling.">
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
<text class="tf-rl-axis-title" x="4" y="34">Attainable TFLOP/s (log)</text>
<text class="tf-rl-axis-title" x="496" y="294" text-anchor="end">Arithmetic intensity: flop per byte (log)</text>
<polyline class="tf-rl-roof" points="52.0,217.9 357.4,67.1 496.0,67.1"/>
<line class="tf-rl-drop" x1="357.4" y1="67.1" x2="357.4" y2="250"/>
<circle class="tf-rl-ridge" cx="357.4" cy="67.1" r="6"/>
<text class="tf-rl-label tf-rl-label--ridge" x="370.4" y="89.1">Ridge point</text>
<text class="tf-rl-sub" x="370.4" y="107.1">14 flop/byte</text>
<circle class="tf-rl-point" cx="135.8" cy="176.5" r="5.5"/>
<text class="tf-rl-label tf-rl-label--point" x="148.8" y="196.5">silu (fp16)</text>
<text class="tf-rl-sub" x="148.8" y="214.5">5 flop / 4 bytes — ceiling 5.1</text>
<text class="tf-rl-roof-label" x="152.5" y="153.3" transform="rotate(-30 152.5 153.3)">bandwidth roof 4.07 TB/s</text>
<text class="tf-rl-roof-label" x="492.0" y="54.1" text-anchor="end">compute roof 57.3 TFLOP/s</text>
</svg>

<figcaption>The bent line is the roofline. Every kernel's performance point lies below it. To the left of the ridge, the ceiling is arithmetic intensity times bandwidth, so attainable compute rises linearly with intensity. To the right, the ceiling is the compute peak, and higher intensity no longer raises it. <code>silu</code> performs 5 operations per 4 bytes moved, for an arithmetic intensity of 1.25 flop/byte, one eleventh of the ridge. Even with bandwidth fully saturated, it can reach only 9% of the compute ceiling.</figcaption>

</figure>

1. [Optimizing Global Memory Access](global-memory-access.md)
2. [Optimizing Shared Memory Access](shared-memory-access.md)
