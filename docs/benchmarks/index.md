# Benchmarks

!!! info "Nightly performance snapshot"
    **GPU:** NVIDIA H200 · **Commit:** [`283f414`](https://github.com/tile-ai/TileOPs/commit/283f414) · **Date:** 2026-06-16 · **179 ops** / 12 families / 1130 configs

**Status**

- Against the strongest competitive baseline where one exists: 🟢 ≥0.95× · 🟡 0.80–0.95× · 🔴 <0.80×
- Against the roofline only where it is reachable (memory-bound ops): 🟢 ≥70% · 🟡 40–70% · 🔴 <40%
- `—` when neither applies
- `torch` is reference only
- **Tests**: ✅ correctness test passed · ❌ failed · `–` no test matched
- **% roof** = achieved ÷ H200 theoretical ceiling at the op's arithmetic intensity

## Attention  <small>(13 ops)</small>

| Op | Tests | Configs | TFLOPS | % roof | vs baseline | Status |
| --- | :---: | ---: | ---: | ---: | --- | :---: |
| [MultiHeadAttentionDecodeWithKVCacheFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/attention/mha.py) | ✅ | 8 | 4.2 | 87% | 🟢 4.19× fa3 | 🟢 |
| [GroupedQueryAttentionBwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/attention/gqa.py) | ✅ | 8 | 114.5 | 12% | 🔴 0.67× fa3 | 🔴 |
| [GroupedQueryAttentionFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/attention/gqa.py) | ✅ | 8 | 223.3 | 23% | 🔴 0.69× fa3 | 🔴 |
| [GroupedQueryAttentionPrefillFP8TensorCoreFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/attention/gqa.py) | – | 8 | 148.8 | – | 🔴 0.23× fa3 | 🔴 |
| [MeanPoolingForward](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/attention/deepseek_nsa.py) | ✅ | 4 | 0.1 | 5% | 🔴 5% roof | 🔴 |
| [MultiHeadAttentionDecodePagedWithKVCacheFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/attention/mha.py) | ✅ | 4 | 0.1 | 2% | 🔴 0.64× flashinfer | 🔴 |
| [DeepSeekSparseAttentionDecodeWithKVCacheFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/attention/deepseek_dsa.py) | ✅ | 2 | 168.3 | 17% | — | — |
| [GroupedQueryAttentionPrefillFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/attention/gqa.py) | – | 6 | 130.8 | 13% | — | — |
| [GroupedQueryAttentionPrefillPagedWithFP8KVCacheFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/attention/gqa.py) | – | 3 | 69.5 | 4% | — | — |
| [GroupedQueryAttentionPrefillPagedWithKVCacheFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/attention/gqa.py) | – | 4 | 120.4 | 12% | — | — |
| [GroupedQueryAttentionPrefillVarlenFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/attention/gqa.py) | – | 3 | 139.7 | 14% | — | — |
| [GroupedQueryAttentionPrefillWithKVCacheFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/attention/gqa.py) | – | 7 | 205.7 | 21% | — | — |
| [MultiHeadLatentAttentionDecodeWithKVCacheFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/attention/deepseek_mla.py) | ✅ | 6 | 188.9 | 19% | — | — |

??? note "Per-config detail"

    | Op | Config | Latency (ms) | TFLOPS | AI | % roof |
    | --- | --- | ---: | ---: | ---: | ---: |
    | DeepSeekSparseAttentionDecodeWithKVCacheFwd | `single-batch-mainstream-float16` | 3.4388 | 169.9 | 1887 | 17% |
    | DeepSeekSparseAttentionDecodeWithKVCacheFwd | `longer-kv-lower-topk-float16` | 0.876 | 166.7 | 981 | 17% |
    | GroupedQueryAttentionBwd | `llama-3.1-8b-short-float16` | 0.1914 | 112.2 | 321 | 11% |
    | GroupedQueryAttentionBwd | `llama-3.1-8b-short-bfloat16` | 0.221 | 97.2 | 324 | 10% |
    | GroupedQueryAttentionBwd | `llama-3.1-8b-long-float16` | 1.4158 | 121.3 | 1348 | 12% |
    | GroupedQueryAttentionBwd | `llama-3.1-8b-long-bfloat16` | 1.3512 | 127.1 | 1271 | 13% |
    | GroupedQueryAttentionBwd | `llama-3.1-70b-short-float16` | 0.2214 | 97.0 | 359 | 10% |
    | GroupedQueryAttentionBwd | `llama-3.1-70b-short-bfloat16` | 0.2015 | 106.6 | 367 | 11% |
    | GroupedQueryAttentionBwd | `llama-3.1-70b-long-float16` | 1.4297 | 120.2 | 1502 | 12% |
    | GroupedQueryAttentionBwd | `llama-3.1-70b-long-bfloat16` | 1.471 | 116.8 | 1460 | 12% |
    | GroupedQueryAttentionFwd | `llama-3.1-8b-short-float16` | 0.0394 | 218.2 | 204 | 22% |
    | GroupedQueryAttentionFwd | `llama-3.1-8b-short-bfloat16` | 0.0423 | 203.2 | 205 | 21% |
    | GroupedQueryAttentionFwd | `llama-3.1-8b-long-float16` | 0.2828 | 243.0 | 810 | 25% |
    | GroupedQueryAttentionFwd | `llama-3.1-8b-long-bfloat16` | 0.3008 | 228.4 | 816 | 23% |
    | GroupedQueryAttentionFwd | `llama-3.1-70b-short-float16` | 0.0472 | 181.8 | 227 | 18% |
    | GroupedQueryAttentionFwd | `llama-3.1-70b-short-bfloat16` | 0.0422 | 203.7 | 226 | 21% |
    | GroupedQueryAttentionFwd | `llama-3.1-70b-long-float16` | 0.2842 | 241.8 | 896 | 24% |
    | GroupedQueryAttentionFwd | `llama-3.1-70b-long-bfloat16` | 0.2553 | 269.2 | 897 | 27% |
    | GroupedQueryAttentionPrefillFP8TensorCoreFwd | `llama-3.1-8b-prefill-fp8tc-bn224-s896-float16` | 0.0879 | 149.6 | – | – |
    | GroupedQueryAttentionPrefillFP8TensorCoreFwd | `llama-3.1-8b-prefill-fp8tc-bn224-s896-bfloat16` | 0.0887 | 148.3 | – | – |
    | GroupedQueryAttentionPrefillFP8TensorCoreFwd | `llama-3.1-8b-prefill-fp8tc-bn224-s1792-float16` | 0.2454 | 214.4 | – | – |
    | GroupedQueryAttentionPrefillFP8TensorCoreFwd | `llama-3.1-8b-prefill-fp8tc-bn224-s1792-bfloat16` | 0.2451 | 214.6 | – | – |
    | GroupedQueryAttentionPrefillFP8TensorCoreFwd | `llama-3.1-70b-prefill-fp8tc-bn224-s3584-float16` | 3.1545 | 133.4 | – | – |
    | GroupedQueryAttentionPrefillFP8TensorCoreFwd | `llama-3.1-70b-prefill-fp8tc-bn224-s3584-bfloat16` | 3.167 | 132.9 | – | – |
    | GroupedQueryAttentionPrefillFP8TensorCoreFwd | `llama-3.1-70b-prefill-fp8tc-bn224-s7168-float16` | 11.2707 | 149.4 | – | – |
    | GroupedQueryAttentionPrefillFP8TensorCoreFwd | `llama-3.1-70b-prefill-fp8tc-bn224-s7168-bfloat16` | 12.2445 | 137.5 | – | – |
    | GroupedQueryAttentionPrefillFwd | `llama-3.1-8b-prefill-dense-float16` | 0.125 | 68.8 | 202 | 7% |
    | GroupedQueryAttentionPrefillFwd | `llama-3.1-8b-prefill-dense-bfloat16` | 0.0523 | 164.6 | 206 | 17% |
    | GroupedQueryAttentionPrefillFwd | `llama-3.1-8b-prefill-dense-q-lt-kv-float16` | 0.4901 | 131.5 | 1315 | 13% |
    | GroupedQueryAttentionPrefillFwd | `llama-3.1-8b-prefill-dense-q-lt-kv-bfloat16` | 0.495 | 130.2 | 1302 | 13% |
    | GroupedQueryAttentionPrefillFwd | `llama-3.1-70b-prefill-dense-q-lt-kv-float16` | 0.4499 | 143.2 | 2046 | 14% |
    | GroupedQueryAttentionPrefillFwd | `llama-3.1-70b-prefill-dense-q-lt-kv-bfloat16` | 1.1763 | 54.8 | 1826 | 6% |
    | GroupedQueryAttentionPrefillPagedWithFP8KVCacheFwd | `qwen35-9b-prefill-paged-fp8-cache-b8-prefix32k-chunk1k-p64-fp16-float16` | 128.6343 | 69.5 | 3472 | 4% |
    | GroupedQueryAttentionPrefillPagedWithFP8KVCacheFwd | `llama31-8b-prefill-paged-fp8-cache-b8-prefix4k-chunk512-p64-fp16-float16` | 2.1016 | 139.0 | 1264 | 7% |
    | GroupedQueryAttentionPrefillPagedWithFP8KVCacheFwd | `gqa-prefill-paged-fp8-cache-softcap50-b4-prefix4k-chunk512-p64-fp16-float16` | 0.2686 | 68.0 | 1359 | 3% |
    | GroupedQueryAttentionPrefillPagedWithKVCacheFwd | `qwen35-9b-prefill-paged-fullattn-b8-prefix32k-chunk1k-p64-partial-rope64-fp16-float16` | 60.6836 | 147.2 | 3680 | 15% |
    | GroupedQueryAttentionPrefillPagedWithKVCacheFwd | `qwen35-9b-prefill-paged-fullattn-mixed-b8-p64-partial-rope64-fp16-float16` | 50.0248 | 66.3 | 2212 | 7% |
    | GroupedQueryAttentionPrefillPagedWithKVCacheFwd | `llama31-8b-prefill-paged-b8-prefix4k-chunk512-p64-full-rope-fp16-float16` | 2.0158 | 144.9 | 1208 | 15% |
    | GroupedQueryAttentionPrefillPagedWithKVCacheFwd | `gqa-prefill-paged-softcap50-b4-prefix4k-chunk512-p64-fp16-float16` | 0.1903 | 96.0 | 1199 | 10% |
    | GroupedQueryAttentionPrefillVarlenFwd | `llama-3.1-8b-prefill-varlen-uniform-fp16` | 0.1236 | 208.7 | 509 | 21% |
    | GroupedQueryAttentionPrefillVarlenFwd | `llama-3.1-8b-prefill-varlen-mixed-fp16` | 0.2035 | 99.0 | 495 | 10% |
    | GroupedQueryAttentionPrefillVarlenFwd | `llama-3.1-70b-prefill-varlen-q-lt-kv-bf16` | 0.3075 | 139.7 | 932 | 14% |
    | GroupedQueryAttentionPrefillWithKVCacheFwd | `llama-3.1-8b-prefill-contig-4k-old-512-new-float16` | 0.1741 | 209.7 | 1234 | 21% |
    | GroupedQueryAttentionPrefillWithKVCacheFwd | `llama-3.1-8b-prefill-contig-4k-old-512-new-bfloat16` | 0.1724 | 211.7 | 1245 | 21% |
    | GroupedQueryAttentionPrefillWithKVCacheFwd | `llama-3.1-8b-prefill-contig-batch2-float16` | 0.0985 | 185.3 | 618 | 19% |
    | GroupedQueryAttentionPrefillWithKVCacheFwd | `llama-3.1-8b-prefill-contig-batch2-bfloat16` | 0.0975 | 187.3 | 624 | 19% |
    | GroupedQueryAttentionPrefillWithKVCacheFwd | `llama-3.1-70b-prefill-contig-4k-old-512-new-float16` | 0.355 | 205.7 | 1870 | 21% |
    | GroupedQueryAttentionPrefillWithKVCacheFwd | `llama-3.1-70b-prefill-contig-4k-old-512-new-bfloat16` | 0.3532 | 206.7 | 1879 | 21% |
    | GroupedQueryAttentionPrefillWithKVCacheFwd | `qwen35-9b-prefill-contig-fullattn-prefix32k-chunk1k-partial-rope64-fp16-float16` | 56.9706 | 19.6 | 1960 | 2% |
    | MeanPoolingForward | `dense-mainstream` | 0.6099 | 0.1 | 0 | 5% |
    | MeanPoolingForward | `dense-batched` | 0.2284 | 0.1 | 0 | 6% |
    | MeanPoolingForward | `varlen-long` | 0.7343 | 0.1 | 0 | 4% |
    | MeanPoolingForward | `varlen-tail` | 0.0276 | 0.3 | 1 | 13% |
    | MultiHeadAttentionDecodePagedWithKVCacheFwd | `single-token-page128-float16` | 0.0277 | 0.1 | 1 | 3% |
    | MultiHeadAttentionDecodePagedWithKVCacheFwd | `batch2-page256-float16` | 0.025 | 0.2 | 2 | 2% |
    | MultiHeadAttentionDecodePagedWithKVCacheFwd | `longer-cache-float16` | 0.0262 | 0.1 | 1 | 2% |
    | MultiHeadAttentionDecodePagedWithKVCacheFwd | `shorter-cache-float16` | 0.0255 | 0.0 | 1 | 1% |
    | MultiHeadAttentionDecodeWithKVCacheFwd | `llama-3.1-8b-4k-float16` | 0.5246 | 4.1 | 1 | 85% |
    | MultiHeadAttentionDecodeWithKVCacheFwd | `llama-3.1-8b-4k-bfloat16` | 0.5267 | 4.1 | 1 | 85% |
    | MultiHeadAttentionDecodeWithKVCacheFwd | `llama-3.1-8b-32k-float16` | 1.0091 | 4.3 | 1 | 89% |
    | MultiHeadAttentionDecodeWithKVCacheFwd | `llama-3.1-8b-32k-bfloat16` | 1.0075 | 4.3 | 1 | 89% |
    | MultiHeadAttentionDecodeWithKVCacheFwd | `llama-3.1-70b-4k-float16` | 0.5305 | 4.0 | 1 | 84% |
    | MultiHeadAttentionDecodeWithKVCacheFwd | `llama-3.1-70b-4k-bfloat16` | 0.5341 | 4.0 | 1 | 84% |
    | MultiHeadAttentionDecodeWithKVCacheFwd | `llama-3.1-70b-32k-float16` | 1.0146 | 4.2 | 1 | 88% |
    | MultiHeadAttentionDecodeWithKVCacheFwd | `llama-3.1-70b-32k-bfloat16` | 1.0116 | 4.2 | 1 | 89% |
    | MultiHeadLatentAttentionDecodeWithKVCacheFwd | `deepseek-v2-4k-float16` | 0.038 | 282.6 | 203 | 29% |
    | MultiHeadLatentAttentionDecodeWithKVCacheFwd | `deepseek-v2-4k-bfloat16` | 0.0379 | 283.0 | 202 | 29% |
    | MultiHeadLatentAttentionDecodeWithKVCacheFwd | `deepseek-v2-32k-float16` | 0.1586 | 135.4 | 212 | 14% |
    | MultiHeadLatentAttentionDecodeWithKVCacheFwd | `deepseek-v2-32k-bfloat16` | 0.1562 | 137.5 | 212 | 14% |
    | MultiHeadLatentAttentionDecodeWithKVCacheFwd | `deepseek-v3-4k-bfloat16` | 0.0224 | 240.2 | 204 | 25% |
    | MultiHeadLatentAttentionDecodeWithKVCacheFwd | `deepseek-v3-32k-bfloat16` | 0.1597 | 67.2 | 210 | 7% |

## Linear Attention / SSM  <small>(16 ops)</small>

| Op | Tests | Configs | TFLOPS | % roof | vs baseline | Status |
| --- | :---: | ---: | ---: | ---: | --- | :---: |
| [Mamba2Fwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/mamba2_fwd.py) | – | 22 | 59.5 | 8% | 🟢 0.95× mamba | 🟢 |
| [SSDChunkScanFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/ssd_chunk_scan.py) | ✅ | 23 | 58.7 | 17% | 🟢 1.13× mamba | 🟢 |
| [SSDChunkStateFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/ssd_chunk_state.py) | ✅ | 10 | 115.6 | 39% | 🟢 1.13× mamba | 🟢 |
| [SSDStatePassingFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/ssd_state_passing.py) | ✅ | 20 | 0.2 | 18% | 🟢 3.40× mamba | 🟢 |
| [SSDDecode](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/ssd_decode.py) | – | 19 | 1.4 | 41% | 🟡 41% roof | 🟡 |
| [DeltaNetBwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/deltanet.py) | – | 10 | 1.2 | 1% | 🔴 0.55× fla | 🔴 |
| [DeltaNetDecode](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/deltanet_recurrence.py) | ✅ | 13 | 0.5 | 8% | 🔴 8% roof | 🔴 |
| [DeltaNetFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/deltanet.py) | – | 12 | 1.4 | 2% | 🔴 0.62× fla | 🔴 |
| [DeltaNet](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/deltanet.py) | – | 10 | 1.5 | 2% | 🔴 0.61× fla | 🔴 |
| [GLABwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/gla.py) | – | 8 | 0.4 | 0% | 🔴 0.75× fla | 🔴 |
| [GLADecode](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/gated_linear_attn.py) | ✅ | 16 | 0.2 | 6% | 🔴 0.29× fla | 🔴 |
| [GLAFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/gla.py) | – | 16 | 0.4 | 1% | 🔴 0.64× fla | 🔴 |
| [GatedDeltaNetBwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/gated_deltanet.py) | – | 10 | 1.0 | 1% | 🔴 0.57× fla | 🔴 |
| [GatedDeltaNetDecode](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/gated_deltanet.py) | ✅ | 13 | 0.8 | 17% | 🔴 0.49× fla | 🔴 |
| [GatedDeltaNetFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/gated_deltanet.py) | – | 12 | 1.3 | 2% | 🔴 0.63× fla | 🔴 |
| [GatedDeltaNet](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/gated_deltanet.py) | – | 10 | 1.3 | 2% | 🔴 0.70× fla | 🔴 |

??? note "Per-config detail"

    | Op | Config | Latency (ms) | TFLOPS | AI | % roof |
    | --- | --- | ---: | ---: | ---: | ---: |
    | DeltaNetBwd | `2-4096-4-64-64-64-dtype0-False` | 0.3897 | 1.4 | 17 | 2% |
    | DeltaNetBwd | `2-4096-4-64-64-32-dtype1-False` | 0.4533 | 1.2 | 17 | 1% |
    | DeltaNetBwd | `2-4096-4-64-64-32-dtype2-False` | 0.4627 | 1.2 | 19 | 1% |
    | DeltaNetBwd | `2-2048-4-64-64-64-dtype3-False` | 0.1421 | 1.9 | 19 | 2% |
    | DeltaNetBwd | `2-8192-4-64-64-64-dtype4-False` | 0.8858 | 1.2 | 17 | 1% |
    | DeltaNetBwd | `2-16384-4-64-64-64-dtype5-False` | 1.8659 | 1.1 | 19 | 1% |
    | DeltaNetBwd | `2-2048-4-64-64-64-dtype6-False` | 0.1389 | 1.9 | 18 | 2% |
    | DeltaNetBwd | `2-4096-4-64-64-64-dtype7-False` | 0.3897 | 1.4 | 17 | 2% |
    | DeltaNetBwd | `2-8192-4-64-64-64-dtype8-False` | 0.8835 | 1.2 | 17 | 1% |
    | DeltaNetBwd | `2-16384-4-64-64-64-dtype9-False` | 1.862 | 1.1 | 19 | 1% |
    | DeltaNetDecode | `1-32-128-128-dtype0` | 0.0123 | 0.3 | 1 | 7% |
    | DeltaNetDecode | `1-32-128-128-dtype1` | 0.0185 | 0.2 | 1 | 2% |
    | DeltaNetDecode | `8-32-128-128-dtype2` | 0.0298 | 0.8 | 1 | 24% |
    | DeltaNetDecode | `8-32-128-128-dtype3` | 0.047 | 0.5 | 2 | 8% |
    | DeltaNetDecode | `16-32-128-128-dtype4` | 0.0724 | 0.7 | 1 | 20% |
    | DeltaNetDecode | `16-32-128-128-dtype5` | 0.1089 | 0.5 | 1 | 6% |
    | DeltaNetDecode | `32-32-128-128-dtype6` | 0.1593 | 0.6 | 1 | 18% |
    | DeltaNetDecode | `32-32-128-128-dtype7` | 0.2386 | 0.4 | 1 | 6% |
    | DeltaNetDecode | `1-32-64-64-dtype8` | 0.0067 | 0.1 | 1 | 3% |
    | DeltaNetDecode | `8-32-64-64-dtype9` | 0.0131 | 0.5 | 1 | 14% |
    | DeltaNetDecode | `32-32-64-64-dtype10` | 0.069 | 0.4 | 1 | 10% |
    | DeltaNetDecode | `64-32-128-128-dtype11` | 0.3423 | 0.6 | 1 | 17% |
    | DeltaNetDecode | `64-32-128-128-dtype12` | 0.4958 | 0.4 | 2 | 6% |
    | DeltaNetFwd | `2-4096-4-64-64-64-dtype0-False` | 0.1141 | 2.4 | 16 | 3% |
    | DeltaNetFwd | `2-4096-4-64-64-32-dtype1-False` | 0.2536 | 1.1 | 15 | 1% |
    | DeltaNetFwd | `2-4096-4-64-64-32-dtype2-False` | 0.2488 | 1.1 | 15 | 1% |
    | DeltaNetFwd | `2-2048-4-64-64-64-dtype3-False` | 0.0623 | 2.1 | 15 | 3% |
    | DeltaNetFwd | `2-8192-4-64-64-64-dtype4-False` | 0.3435 | 1.6 | 16 | 2% |
    | DeltaNetFwd | `2-16384-4-64-64-64-dtype5-False` | 0.8157 | 1.3 | 16 | 2% |
    | DeltaNetFwd | `2-32768-4-64-64-64-dtype6-False` | 1.7166 | 1.2 | 16 | 2% |
    | DeltaNetFwd | `2-2048-4-64-64-64-dtype7-False` | 0.0628 | 2.1 | 16 | 3% |
    | DeltaNetFwd | `2-4096-4-64-64-64-dtype8-False` | 0.1147 | 2.3 | 16 | 3% |
    | DeltaNetFwd | `2-8192-4-64-64-64-dtype9-False` | 0.3488 | 1.5 | 15 | 2% |
    | DeltaNetFwd | `2-16384-4-64-64-64-dtype10-False` | 0.8123 | 1.3 | 16 | 2% |
    | DeltaNetFwd | `2-32768-4-64-64-64-dtype11-False` | 1.7141 | 1.2 | 16 | 2% |
    | DeltaNet | `2-4096-4-64-64-64-dtype0-False` | 0.3768 | 2.1 | 18 | 2% |
    | DeltaNet | `2-4096-4-64-64-32-dtype1-False` | 0.5589 | 1.4 | 18 | 2% |
    | DeltaNet | `2-4096-4-64-64-32-dtype2-False` | 0.5712 | 1.4 | 18 | 2% |
    | DeltaNet | `2-2048-4-64-64-64-dtype3-False` | 0.1949 | 2.1 | 17 | 2% |
    | DeltaNet | `2-8192-4-64-64-64-dtype4-False` | 1.0865 | 1.5 | 16 | 2% |
    | DeltaNet | `2-16384-4-64-64-64-dtype5-False` | 2.5357 | 1.3 | 18 | 1% |
    | DeltaNet | `2-2048-4-64-64-64-dtype6-False` | 0.1949 | 2.1 | 17 | 2% |
    | DeltaNet | `2-4096-4-64-64-64-dtype7-False` | 0.3775 | 2.1 | 18 | 2% |
    | DeltaNet | `2-8192-4-64-64-64-dtype8-False` | 1.1012 | 1.5 | 18 | 2% |
    | DeltaNet | `2-16384-4-64-64-64-dtype9-False` | 2.5542 | 1.3 | 18 | 1% |
    | GLABwd | `2-2048-4-64-64-64-dtype0-False` | 0.7381 | 0.4 | 18 | 0% |
    | GLABwd | `2-4096-4-64-64-64-dtype1-False` | 1.4303 | 0.4 | 19 | 0% |
    | GLABwd | `2-8192-4-64-64-64-dtype2-False` | 2.9331 | 0.4 | 18 | 0% |
    | GLABwd | `2-16384-4-64-64-64-dtype3-False` | 6.0216 | 0.4 | 18 | 0% |
    | GLABwd | `2-2048-4-64-64-64-dtype4-False` | 0.6871 | 0.4 | 20 | 0% |
    | GLABwd | `2-4096-4-64-64-64-dtype5-False` | 1.5052 | 0.4 | 18 | 0% |
    | GLABwd | `2-8192-4-64-64-64-dtype6-False` | 3.1178 | 0.3 | 17 | 0% |
    | GLABwd | `2-16384-4-64-64-64-dtype7-False` | 6.1299 | 0.3 | 18 | 0% |
    | GLADecode | `1-32-64-64-dtype0` | 0.01 | 0.1 | 0 | 2% |
    | GLADecode | `1-32-128-128-dtype1` | 0.0202 | 0.1 | 0 | 4% |
    | GLADecode | `1-32-128-128-dtype2` | 0.0211 | 0.1 | 1 | 2% |
    | GLADecode | `1-32-128-128-dtype3` | 0.0201 | 0.1 | 1 | 2% |
    | GLADecode | `8-32-128-128-dtype4` | 0.0507 | 0.3 | 0 | 14% |
    | GLADecode | `8-32-128-128-dtype5` | 0.0518 | 0.3 | 1 | 7% |
    | GLADecode | `8-32-128-128-dtype6` | 0.0474 | 0.4 | 1 | 8% |
    | GLADecode | `16-32-128-128-dtype7` | 0.1252 | 0.3 | 0 | 11% |
    | GLADecode | `16-32-128-128-dtype8` | 0.1268 | 0.3 | 1 | 6% |
    | GLADecode | `16-32-128-128-dtype9` | 0.1198 | 0.3 | 1 | 6% |
    | GLADecode | `32-32-128-128-dtype10` | 0.2734 | 0.2 | 0 | 10% |
    | GLADecode | `32-32-128-128-dtype11` | 0.2855 | 0.2 | 1 | 5% |
    | GLADecode | `32-32-128-128-dtype12` | 0.266 | 0.2 | 1 | 5% |
    | GLADecode | `64-32-128-128-dtype13` | 0.5649 | 0.2 | 0 | 10% |
    | GLADecode | `64-32-128-128-dtype14` | 0.5836 | 0.2 | 1 | 5% |
    | GLADecode | `64-32-128-128-dtype15` | 0.548 | 0.2 | 1 | 5% |
    | GLAFwd | `2-2048-4-64-64-64-dtype0-False` | 0.1211 | 1.1 | 16 | 1% |
    | GLAFwd | `2-4096-4-64-64-64-dtype1-False` | 0.5924 | 0.5 | 15 | 1% |
    | GLAFwd | `2-8192-4-64-64-64-dtype2-False` | 1.3378 | 0.4 | 13 | 1% |
    | GLAFwd | `2-16384-4-64-64-64-dtype3-False` | 2.7387 | 0.4 | 20 | 0% |
    | GLAFwd | `2-2048-4-64-64-64-dtype4-False` | 0.2598 | 0.5 | 17 | 1% |
    | GLAFwd | `2-4096-4-64-64-64-dtype5-False` | 0.4308 | 0.6 | 16 | 1% |
    | GLAFwd | `2-8192-4-64-64-64-dtype6-False` | 1.324 | 0.4 | 14 | 1% |
    | GLAFwd | `2-16384-4-64-64-64-dtype7-False` | 2.7851 | 0.4 | 20 | 0% |
    | GLAFwd | `2-2048-4-64-64-64-dtype0-False` | 1.0668 | 0.4 | 19 | 0% |
    | GLAFwd | `2-4096-4-64-64-64-dtype1-False` | 2.0786 | 0.4 | 20 | 0% |
    | GLAFwd | `2-8192-4-64-64-64-dtype2-False` | 4.3046 | 0.4 | 18 | 0% |
    | GLAFwd | `2-16384-4-64-64-64-dtype3-False` | 8.7455 | 0.4 | 18 | 0% |
    | GLAFwd | `2-2048-4-64-64-64-dtype4-False` | 1.0944 | 0.4 | 18 | 0% |
    | GLAFwd | `2-4096-4-64-64-64-dtype5-False` | 2.0 | 0.4 | 20 | 0% |
    | GLAFwd | `2-8192-4-64-64-64-dtype6-False` | 1.8729 | 0.9 | 17 | 1% |
    | GLAFwd | `2-16384-4-64-64-64-dtype7-False` | 3.9321 | 0.8 | 16 | 1% |
    | GatedDeltaNetBwd | `2-4096-4-64-64-32-dtype0-False` | 0.5747 | 0.9 | 19 | 1% |
    | GatedDeltaNetBwd | `2-4096-4-64-64-32-dtype1-False` | 0.5364 | 1.0 | 17 | 1% |
    | GatedDeltaNetBwd | `2-2048-4-64-64-64-dtype2-False` | 0.22 | 1.2 | 17 | 1% |
    | GatedDeltaNetBwd | `2-4096-4-64-64-64-dtype3-False` | 0.415 | 1.3 | 18 | 1% |
    | GatedDeltaNetBwd | `2-8192-4-64-64-64-dtype4-False` | 1.0131 | 1.1 | 18 | 1% |
    | GatedDeltaNetBwd | `2-16384-4-64-64-64-dtype5-False` | 2.5484 | 0.8 | 17 | 1% |
    | GatedDeltaNetBwd | `2-2048-4-64-64-64-dtype6-False` | 0.2227 | 1.2 | 17 | 1% |
    | GatedDeltaNetBwd | `2-4096-4-64-64-64-dtype7-False` | 0.4218 | 1.3 | 18 | 1% |
    | GatedDeltaNetBwd | `2-8192-4-64-64-64-dtype8-False` | 1.0563 | 1.0 | 17 | 1% |
    | GatedDeltaNetBwd | `2-16384-4-64-64-64-dtype9-False` | 2.6454 | 0.8 | 20 | 1% |
    | GatedDeltaNetDecode | `1-32-128-128-dtype0` | 0.0124 | 0.2 | 1 | 7% |
    | GatedDeltaNetDecode | `1-32-128-128-dtype1` | 0.0041 | 0.8 | 1 | 11% |
    | GatedDeltaNetDecode | `8-32-128-128-dtype2` | 0.0297 | 0.8 | 1 | 24% |
    | GatedDeltaNetDecode | `8-32-128-128-dtype3` | 0.0093 | 2.7 | 1 | 38% |
    | GatedDeltaNetDecode | `16-32-128-128-dtype4` | 0.065 | 0.8 | 1 | 22% |
    | GatedDeltaNetDecode | `16-32-128-128-dtype5` | 0.0167 | 3.0 | 1 | 42% |
    | GatedDeltaNetDecode | `32-32-128-128-dtype6` | 0.1644 | 0.6 | 1 | 17% |
    | GatedDeltaNetDecode | `32-32-128-128-dtype7` | 0.0385 | 2.6 | 1 | 37% |
    | GatedDeltaNetDecode | `1-32-64-64-dtype8` | 0.0069 | 0.1 | 1 | 3% |
    | GatedDeltaNetDecode | `8-32-64-64-dtype9` | 0.0136 | 0.5 | 1 | 13% |
    | GatedDeltaNetDecode | `32-32-64-64-dtype10` | 0.072 | 0.3 | 1 | 10% |
    | GatedDeltaNetDecode | `64-32-128-128-dtype11` | 0.3445 | 0.6 | 1 | 16% |
    | GatedDeltaNetDecode | `64-32-128-128-dtype12` | 0.0808 | 2.5 | 1 | 35% |
    | GatedDeltaNetFwd | `2-4096-4-64-64-32-dtype0-False` | 0.238 | 1.1 | 16 | 1% |
    | GatedDeltaNetFwd | `2-4096-4-64-64-32-dtype1-False` | 0.2384 | 1.1 | 16 | 1% |
    | GatedDeltaNetFwd | `2-2048-4-64-64-64-dtype2-False` | 0.0864 | 1.6 | 16 | 2% |
    | GatedDeltaNetFwd | `2-4096-4-64-64-64-dtype3-False` | 0.1512 | 1.8 | 16 | 2% |
    | GatedDeltaNetFwd | `2-8192-4-64-64-64-dtype4-False` | 0.3551 | 1.5 | 15 | 2% |
    | GatedDeltaNetFwd | `2-16384-4-64-64-64-dtype5-False` | 0.9145 | 1.2 | 17 | 1% |
    | GatedDeltaNetFwd | `2-32768-4-64-64-64-dtype6-False` | 2.0017 | 1.1 | 15 | 1% |
    | GatedDeltaNetFwd | `2-2048-4-64-64-64-dtype7-False` | 0.0904 | 1.5 | 17 | 2% |
    | GatedDeltaNetFwd | `2-4096-4-64-64-64-dtype8-False` | 0.1703 | 1.6 | 16 | 2% |
    | GatedDeltaNetFwd | `2-8192-4-64-64-64-dtype9-False` | 0.3943 | 1.4 | 15 | 2% |
    | GatedDeltaNetFwd | `2-16384-4-64-64-64-dtype10-False` | 0.9819 | 1.1 | 16 | 1% |
    | GatedDeltaNetFwd | `2-32768-4-64-64-64-dtype11-False` | 2.1581 | 1.0 | 17 | 1% |
    | GatedDeltaNet | `2-4096-4-64-64-32-dtype0-False` | 0.7377 | 1.1 | 18 | 1% |
    | GatedDeltaNet | `2-4096-4-64-64-32-dtype1-False` | 0.7389 | 1.1 | 18 | 1% |
    | GatedDeltaNet | `2-2048-4-64-64-64-dtype2-False` | 0.3013 | 1.3 | 17 | 2% |
    | GatedDeltaNet | `2-4096-4-64-64-64-dtype3-False` | 0.5639 | 1.4 | 18 | 2% |
    | GatedDeltaNet | `2-8192-4-64-64-64-dtype4-False` | 1.1565 | 1.4 | 17 | 2% |
    | GatedDeltaNet | `2-16384-4-64-64-64-dtype5-False` | 3.5283 | 0.9 | 18 | 1% |
    | GatedDeltaNet | `2-2048-4-64-64-64-dtype6-False` | 0.3107 | 1.3 | 19 | 1% |
    | GatedDeltaNet | `2-4096-4-64-64-64-dtype7-False` | 0.5918 | 1.4 | 17 | 2% |
    | GatedDeltaNet | `2-8192-4-64-64-64-dtype8-False` | 1.2015 | 1.3 | 17 | 2% |
    | GatedDeltaNet | `2-16384-4-64-64-64-dtype9-False` | 3.5203 | 0.9 | 18 | 1% |
    | Mamba2Fwd | `smoke-b1-s256-4h` | 0.0356 | 1.9 | 188 | 0% |
    | Mamba2Fwd | `smoke-b2-s512-8h` | 0.0401 | 13.4 | 149 | 2% |
    | Mamba2Fwd | `full-130m-latency` | 0.0634 | 50.8 | 159 | 7% |
    | Mamba2Fwd | `full-130m-serving` | 0.3101 | 83.2 | 160 | 11% |
    | Mamba2Fwd | `full-130m-throughput` | 1.7598 | 58.6 | 158 | 8% |
    | Mamba2Fwd | `full-130m-long-ctx` | 3.8369 | 53.8 | 158 | 7% |
    | Mamba2Fwd | `full-370m-latency` | 0.0885 | 72.8 | 165 | 9% |
    | Mamba2Fwd | `full-370m-serving` | 0.552 | 93.4 | 164 | 12% |
    | Mamba2Fwd | `full-370m-throughput` | 3.497 | 59.0 | 164 | 8% |
    | Mamba2Fwd | `full-370m-long-ctx` | 7.4573 | 55.3 | 163 | 7% |
    | Mamba2Fwd | `full-780m-latency` | 0.106 | 81.0 | 165 | 10% |
    | Mamba2Fwd | `full-780m-serving` | 0.8401 | 81.8 | 167 | 10% |
    | Mamba2Fwd | `full-780m-throughput` | 4.6146 | 59.6 | 166 | 8% |
    | Mamba2Fwd | `full-780m-long-ctx` | 9.6371 | 57.1 | 168 | 7% |
    | Mamba2Fwd | `full-1.3b-latency` | 0.1293 | 83.1 | 166 | 10% |
    | Mamba2Fwd | `full-1.3b-serving` | 0.8336 | 103.1 | 166 | 13% |
    | Mamba2Fwd | `full-1.3b-throughput` | 5.8167 | 59.1 | 164 | 8% |
    | Mamba2Fwd | `full-1.3b-long-ctx` | 12.1443 | 56.6 | 167 | 7% |
    | Mamba2Fwd | `full-2.7b-latency` | 0.1908 | 90.1 | 167 | 11% |
    | Mamba2Fwd | `full-2.7b-serving` | 1.66 | 82.8 | 166 | 10% |
    | Mamba2Fwd | `full-2.7b-throughput` | 9.221 | 59.7 | 166 | 8% |
    | Mamba2Fwd | `full-2.7b-long-ctx` | 18.5093 | 59.4 | 165 | 8% |
    | SSDChunkScanFwd | `b1-c2-L64-h4-p64-n32-fp16` | 0.0059 | 0.7 | 18 | 1% |
    | SSDChunkScanFwd | `b2-c4-L64-h8-p64-n64-fp16` | 0.0069 | 7.4 | 21 | 7% |
    | SSDChunkScanFwd | `b1-c2-L128-h4-p128-n32-bf16` | 0.0079 | 3.2 | 27 | 2% |
    | SSDChunkScanFwd | `b2-c2-L64-h4-p64-n32-bf16` | 0.006 | 1.4 | 16 | 2% |
    | SSDChunkScanFwd | `latency-130m-4k` | 0.0422 | 76.5 | 68 | 24% |
    | SSDChunkScanFwd | `serving-130m-4k` | 0.7108 | 36.3 | 67 | 11% |
    | SSDChunkScanFwd | `longctx-130m-32k` | 1.6912 | 61.1 | 68 | 19% |
    | SSDChunkScanFwd | `latency-370m-4k` | 0.0648 | 66.4 | 68 | 20% |
    | SSDChunkScanFwd | `serving-370m-4k` | 0.5579 | 61.7 | 69 | 19% |
    | SSDChunkScanFwd | `longctx-370m-32k` | 2.1774 | 63.2 | 69 | 19% |
    | SSDChunkScanFwd | `throughput-370m-2k` | 1.0918 | 63.1 | 69 | 19% |
    | SSDChunkScanFwd | `latency-780m-4k` | 0.0836 | 77.2 | 70 | 23% |
    | SSDChunkScanFwd | `serving-780m-4k` | 1.1175 | 46.2 | 70 | 14% |
    | SSDChunkScanFwd | `longctx-780m-32k` | 3.3481 | 61.7 | 70 | 18% |
    | SSDChunkScanFwd | `throughput-780m-2k` | 0.8407 | 61.4 | 70 | 18% |
    | SSDChunkScanFwd | `latency-1p3b-4k` | 0.1604 | 53.7 | 71 | 16% |
    | SSDChunkScanFwd | `serving-1p3b-4k` | 1.7108 | 40.2 | 71 | 12% |
    | SSDChunkScanFwd | `longctx-1p3b-32k` | 2.5751 | 53.5 | 70 | 16% |
    | SSDChunkScanFwd | `throughput-1p3b-2k` | 0.5595 | 61.5 | 71 | 18% |
    | SSDChunkScanFwd | `latency-2p7b-4k` | 0.1924 | 55.9 | 71 | 16% |
    | SSDChunkScanFwd | `serving-2p7b-4k` | 0.7101 | 60.6 | 70 | 18% |
    | SSDChunkScanFwd | `longctx-2p7b-32k` | 3.3252 | 51.8 | 71 | 15% |
    | SSDChunkScanFwd | `throughput-2p7b-2k` | 0.3667 | 58.7 | 71 | 17% |
    | SSDChunkStateFwd | `debug-b1-c2-L64-h4-p64-n32-g1-fp16` | 0.0032 | 0.7 | 16 | 1% |
    | SSDChunkStateFwd | `mamba2-370m-b1-L2k-fp16` | 0.0103 | 104.4 | 60 | 36% |
    | SSDChunkStateFwd | `mamba2-370m-b4-L2k-fp16` | 0.0339 | 126.8 | 60 | 44% |
    | SSDChunkStateFwd | `mamba2-1p3b-b1-L2k-fp16` | 0.0183 | 117.3 | 61 | 40% |
    | SSDChunkStateFwd | `mamba2-1p3b-b4-L2k-fp16` | 0.0703 | 122.1 | 61 | 42% |
    | SSDChunkStateFwd | `mamba2-2p7b-b1-L2k-bf16` | 0.0236 | 113.8 | 61 | 39% |
    | SSDChunkStateFwd | `mamba2-2p7b-b4-L2k-bf16` | 0.0968 | 110.9 | 61 | 38% |
    | SSDChunkStateFwd | `mamba2-1p3b-b1-L8k-fp16` | 0.063 | 136.4 | 61 | 46% |
    | SSDChunkStateFwd | `mamba2-1p3b-b1-L2k-seqidx-fp16` | 0.0201 | 106.8 | 61 | 36% |
    | SSDChunkStateFwd | `mamba2-1p3b-b4-L2k-seqidx-fp16` | 0.0683 | 125.8 | 61 | 43% |
    | SSDDecode | `b1-h4-p64-n16-g1-fp16` | 0.0022 | 0.0 | 0 | 0% |
    | SSDDecode | `b2-h8-p64-n32-g2-fp16` | 0.0024 | 0.1 | 1 | 3% |
    | SSDDecode | `b1-h4-p64-n16-g1-bf16` | 0.0022 | 0.0 | 0 | 0% |
    | SSDDecode | `b2-h8-p128-n64-g4-bf16` | 0.0028 | 0.3 | 1 | 10% |
    | SSDDecode | `latency-130m` | 0.0031 | 0.4 | 0 | 16% |
    | SSDDecode | `serving-130m` | 0.0069 | 1.4 | 1 | 41% |
    | SSDDecode | `throughput-130m` | 0.0368 | 2.0 | 1 | 58% |
    | SSDDecode | `latency-370m` | 0.0033 | 0.5 | 0 | 20% |
    | SSDDecode | `serving-370m` | 0.0089 | 1.4 | 1 | 42% |
    | SSDDecode | `throughput-370m` | 0.0485 | 2.1 | 1 | 59% |
    | SSDDecode | `latency-780m` | 0.0038 | 0.6 | 0 | 26% |
    | SSDDecode | `serving-780m` | 0.0111 | 1.7 | 1 | 51% |
    | SSDDecode | `throughput-780m` | 0.0393 | 1.9 | 1 | 55% |
    | SSDDecode | `latency-1p3b` | 0.0047 | 0.7 | 0 | 28% |
    | SSDDecode | `serving-1p3b` | 0.014 | 1.8 | 1 | 54% |
    | SSDDecode | `throughput-1p3b` | 0.0252 | 2.0 | 1 | 58% |
    | SSDDecode | `latency-2p7b` | 0.005 | 0.8 | 0 | 33% |
    | SSDDecode | `serving-2p7b` | 0.0097 | 1.6 | 1 | 51% |
    | SSDDecode | `throughput-2p7b` | 0.0166 | 1.9 | 1 | 56% |
    | SSDStatePassingFwd | `b1-c2-h4-d32-fp16` | 0.0018 | – | – | – |
    | SSDStatePassingFwd | `b2-c4-h8-d64-fp16` | 0.0019 | – | – | – |
    | SSDStatePassingFwd | `b1-c2-h4-d32-bf16` | 0.0018 | – | – | – |
    | SSDStatePassingFwd | `b2-c4-h8-d64-bf16` | 0.0019 | – | – | – |
    | SSDStatePassingFwd | `latency-130m-4k` | 0.0023 | 0.0 | 0 | 3% |
    | SSDStatePassingFwd | `serving-130m-4k` | 0.0033 | 0.2 | 0 | 16% |
    | SSDStatePassingFwd | `longctx-130m-32k` | 0.0112 | 0.3 | 0 | 18% |
    | SSDStatePassingFwd | `latency-370m-4k` | 0.0022 | 0.1 | 0 | 4% |
    | SSDStatePassingFwd | `serving-370m-4k` | 0.0029 | 0.4 | 0 | 25% |
    | SSDStatePassingFwd | `longctx-370m-32k` | 0.0102 | 0.4 | 0 | 26% |
    | SSDStatePassingFwd | `throughput-370m-2k` | 0.0088 | 0.2 | 0 | 18% |
    | SSDStatePassingFwd | `latency-780m-4k` | 0.0023 | 0.1 | 0 | 6% |
    | SSDStatePassingFwd | `serving-780m-4k` | 0.0033 | 0.5 | 0 | 32% |
    | SSDStatePassingFwd | `longctx-780m-32k` | 0.0213 | 0.3 | 0 | 19% |
    | SSDStatePassingFwd | `latency-1p3b-4k` | 0.0024 | 0.1 | 0 | 8% |
    | SSDStatePassingFwd | `serving-1p3b-4k` | 0.0057 | 0.4 | 0 | 25% |
    | SSDStatePassingFwd | `longctx-1p3b-32k` | 0.0101 | 0.4 | 0 | 26% |
    | SSDStatePassingFwd | `latency-2p7b-4k` | 0.0024 | 0.1 | 0 | 9% |
    | SSDStatePassingFwd | `serving-2p7b-4k` | 0.0032 | 0.4 | 0 | 28% |
    | SSDStatePassingFwd | `longctx-2p7b-32k` | 0.0107 | 0.5 | 0 | 31% |

## Scan  <small>(1 ops)</small>

| Op | Tests | Configs | TFLOPS | % roof | vs baseline | Status |
| --- | :---: | ---: | ---: | ---: | --- | :---: |
| [DaCumsumFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/da_cumsum.py) | ✅ | 7 | – | 1% | 🟢 1.05× mamba | 🟢 |

??? note "Per-config detail"

    | Op | Config | Latency (ms) | TFLOPS | AI | % roof |
    | --- | --- | ---: | ---: | ---: | ---: |
    | DaCumsumFwd | `1-2-64-4-False-False-False` | 0.002 | – | – | – |
    | DaCumsumFwd | `1-2-64-4-True-False-False` | 0.002 | – | – | – |
    | DaCumsumFwd | `1-2-64-4-False-True-False` | 0.0021 | – | – | – |
    | DaCumsumFwd | `1-2-64-4-True-True-False` | 0.0025 | – | – | – |
    | DaCumsumFwd | `2-4-64-8-False-False-False` | 0.0021 | 0.0 | 0 | 0% |
    | DaCumsumFwd | `1-2-128-4-False-False-False` | 0.0022 | – | – | – |
    | DaCumsumFwd | `2-4-128-16-True-True-False` | 0.0026 | 0.1 | 1 | 2% |

## Normalization  <small>(7 ops)</small>

| Op | Tests | Configs | TFLOPS | % roof | vs baseline | Status |
| --- | :---: | ---: | ---: | ---: | --- | :---: |
| [FusedAddRMSNormFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/norm/fused_add_rms_norm.py) | ✅ | 6 | 1.9 | 65% | 🟡 65% roof | 🟡 |
| [LayerNormFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/norm/layer_norm.py) | ✅ | 9 | 2.9 | 49% | 🟡 49% roof | 🟡 |
| [RMSNormFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/norm/rms_norm.py) | ✅ | 9 | 2.6 | 55% | 🟡 55% roof | 🟡 |
| [AdaLayerNormFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/norm/ada_layer_norm.py) | ✅ | 5 | 0.0 | 22% | 🔴 22% roof | 🔴 |
| [AdaLayerNormZeroFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/norm/ada_layer_norm_zero.py) | ✅ | 5 | 0.0 | 24% | 🔴 24% roof | 🔴 |
| [GroupNormFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/norm/group_norm.py) | ✅ | 3 | 0.3 | 5% | 🔴 5% roof | 🔴 |
| [GroupNormFwdNoAffine](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/norm/group_norm.py) | – | 4 | 0.3 | 16% | — | — |

??? note "Per-config detail"

    | Op | Config | Latency (ms) | TFLOPS | AI | % roof |
    | --- | --- | ---: | ---: | ---: | ---: |
    | AdaLayerNormFwd | `dit-xl-2-float16` | 0.7616 | 0.0 | 1 | 0% |
    | AdaLayerNormFwd | `dit-xl-2-bfloat16` | 0.9558 | 0.0 | 1 | 0% |
    | AdaLayerNormFwd | `llama-3.1-8b-prefill-float16` | 0.0318 | 1.3 | 1 | 44% |
    | AdaLayerNormFwd | `llama-3.1-8b-prefill-bfloat16` | 0.0322 | 1.3 | 1 | 43% |
    | AdaLayerNormFwd | `llama-3.1-8b-decode-bfloat16` | 0.0054 | – | – | – |
    | AdaLayerNormZeroFwd | `dit-xl-2-float16` | 0.7616 | 0.0 | 0 | 0% |
    | AdaLayerNormZeroFwd | `dit-xl-2-bfloat16` | 1.1352 | 0.0 | 1 | 0% |
    | AdaLayerNormZeroFwd | `llama-3.1-8b-prefill-float16` | 0.0362 | 1.4 | 1 | 48% |
    | AdaLayerNormZeroFwd | `llama-3.1-8b-prefill-bfloat16` | 0.0357 | 1.4 | 1 | 49% |
    | AdaLayerNormZeroFwd | `llama-3.1-8b-decode-bfloat16` | 0.0058 | – | – | – |
    | FusedAddRMSNormFwd | `llama-3.1-8b-prefill-float16` | 0.0215 | 1.9 | 1 | 65% |
    | FusedAddRMSNormFwd | `llama-3.1-8b-prefill-bfloat16` | 0.0215 | 1.9 | 1 | 65% |
    | FusedAddRMSNormFwd | `llama-3.1-8b-decode-bfloat16` | 0.0032 | 0.0 | 1 | 0% |
    | FusedAddRMSNormFwd | `llama-3.1-70b-prefill-float16` | 0.0397 | 2.1 | 1 | 71% |
    | FusedAddRMSNormFwd | `llama-3.1-70b-prefill-bfloat16` | 0.0432 | 1.9 | 1 | 65% |
    | FusedAddRMSNormFwd | `llama-3.1-70b-decode-bfloat16` | 0.004 | 0.0 | 0 | 0% |
    | GroupNormFwd | `image-g32-float16` | 0.0177 | 0.3 | 1 | 5% |
    | GroupNormFwd | `image-g32-bfloat16` | 0.0176 | 0.3 | 1 | 5% |
    | GroupNormFwd | `wider-channel-g32-float16` | 3.396 | – | – | – |
    | GroupNormFwdNoAffine | `image-g32-float16` | 0.0057 | 0.6 | 1 | 15% |
    | GroupNormFwdNoAffine | `image-g32-bfloat16` | 0.0056 | 0.6 | 1 | 16% |
    | GroupNormFwdNoAffine | `wider-channel-g32-float16` | 2.9737 | – | – | – |
    | GroupNormFwdNoAffine | `tail-spatial-g16-float16` | 3.2808 | – | – | – |
    | LayerNormFwd | `llama-3.1-8b-prefill-float16` | 0.0144 | 2.9 | 1 | 49% |
    | LayerNormFwd | `llama-3.1-8b-prefill-bfloat16` | 0.0152 | 2.8 | 1 | 46% |
    | LayerNormFwd | `llama-3.1-8b-decode-bfloat16` | 0.0036 | 0.0 | 1 | 0% |
    | LayerNormFwd | `llama-3.1-70b-prefill-float16` | 0.0267 | 3.1 | 1 | 52% |
    | LayerNormFwd | `llama-3.1-70b-prefill-bfloat16` | 0.0271 | 3.1 | 1 | 52% |
    | LayerNormFwd | `llama-3.1-70b-decode-bfloat16` | 0.0038 | 0.0 | 0 | 0% |
    | LayerNormFwd | `llama-3.1-405b-prefill-float16` | 0.0515 | 3.3 | 1 | 54% |
    | LayerNormFwd | `llama-3.1-405b-prefill-bfloat16` | 0.0522 | 3.2 | 1 | 54% |
    | LayerNormFwd | `llama-3.1-405b-decode-bfloat16` | 0.0049 | 0.0 | 1 | 1% |
    | RMSNormFwd | `llama-3.1-8b-prefill-float16` | 0.0129 | 2.6 | 1 | 54% |
    | RMSNormFwd | `llama-3.1-8b-prefill-bfloat16` | 0.0128 | 2.6 | 1 | 55% |
    | RMSNormFwd | `llama-3.1-8b-decode-bfloat16` | 0.0027 | 0.0 | 1 | 0% |
    | RMSNormFwd | `llama-3.1-70b-prefill-float16` | 0.0224 | 3.0 | 1 | 62% |
    | RMSNormFwd | `llama-3.1-70b-prefill-bfloat16` | 0.0231 | 2.9 | 1 | 61% |
    | RMSNormFwd | `llama-3.1-70b-decode-bfloat16` | 0.0032 | 0.0 | 0 | 0% |
    | RMSNormFwd | `llama-3.1-405b-prefill-float16` | 0.0432 | 3.1 | 1 | 65% |
    | RMSNormFwd | `llama-3.1-405b-prefill-bfloat16` | 0.0442 | 3.0 | 1 | 63% |
    | RMSNormFwd | `llama-3.1-405b-decode-bfloat16` | 0.0045 | 0.0 | 0 | 0% |

## Mixture of Experts  <small>(6 ops)</small>

| Op | Tests | Configs | TFLOPS | % roof | vs baseline | Status |
| --- | :---: | ---: | ---: | ---: | --- | :---: |
| [FusedMoEExpertsNopadPersistent3WGFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/moe/routed_expert/fused_routed_expert.py) | – | 6 | 106.9 | 31% | 🟢 1.01× vllm-triton | 🟢 |
| [FusedMoeFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/moe/fused_moe.py) | – | 4 | 111.0 | 31% | 🟢 0.95× vllm | 🟢 |
| [MoePermuteNopadFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/moe/routed_expert/permute_nopad.py) | – | 16 | – | – | 🟢 1.47× vllm | 🟢 |
| [moe_unpermute](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+moe_unpermute&type=code) | – | 8 | 1.2 | 29% | 🟢 1.45× vllm | 🟢 |
| [FusedMoeFwdCbFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/moe/fused_moe.py) | – | 2 | 108.7 | 42% | 🟡 42% roof | 🟡 |
| [MoeGroupedGemmNopadFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/moe/routed_expert/moe_grouped_gemm_nopad.py) | – | 4 | 78.6 | 31% | 🔴 31% roof | 🔴 |

??? note "Per-config detail"

    | Op | Config | Latency (ms) | TFLOPS | AI | % roof |
    | --- | --- | ---: | ---: | ---: | ---: |
    | FusedMoEExpertsNopadPersistent3WGFwd | `qwen3-235b-decode-bfloat16` | 6.7509 | 53.4 | 32 | 35% |
    | FusedMoEExpertsNopadPersistent3WGFwd | `qwen3-235b-prefill-bfloat16` | 13.2118 | 218.5 | 254 | 22% |
    | FusedMoEExpertsNopadPersistent3WGFwd | `deepseek-v3-decode-bfloat16` | 13.2467 | 27.2 | 16 | 35% |
    | FusedMoEExpertsNopadPersistent3WGFwd | `deepseek-v3-prefill-bfloat16` | 18.0015 | 160.3 | 127 | 26% |
    | FusedMoEExpertsNopadPersistent3WGFwd | `prefill-4096` | 0.2226 | 347.3 | 709 | 35% |
    | FusedMoEExpertsNopadPersistent3WGFwd | `decode-128` | 0.061 | 39.6 | 31 | 26% |
    | FusedMoeFwdCbFwd | `kimi-k2-decode-bfloat16` | 8.3492 | 43.2 | 16 | 56% |
    | FusedMoeFwdCbFwd | `kimi-k2-prefill-bfloat16` | 16.5623 | 174.3 | 127 | 29% |
    | FusedMoeFwd | `qwen3-235b-decode-bfloat16` | 6.1349 | 58.8 | 32 | 38% |
    | FusedMoeFwd | `qwen3-235b-prefill-bfloat16` | 13.3335 | 216.5 | 255 | 22% |
    | FusedMoeFwd | `deepseek-v3-decode-bfloat16` | 13.335 | 27.1 | 16 | 35% |
    | FusedMoeFwd | `deepseek-v3-prefill-bfloat16` | 17.6933 | 163.1 | 127 | 27% |
    | MoeGroupedGemmNopadFwd | `deepseek-v3-decode-gate-up-bfloat16` | 8.0325 | 29.9 | 16 | 39% |
    | MoeGroupedGemmNopadFwd | `deepseek-v3-prefill-gate-up-bfloat16` | 15.1254 | 127.2 | 122 | 22% |
    | MoeGroupedGemmNopadFwd | `deepseek-v3-decode-down-bfloat16` | 4.0111 | 30.0 | 16 | 39% |
    | MoeGroupedGemmNopadFwd | `deepseek-v3-prefill-down-bfloat16` | 7.5367 | 127.7 | 118 | 23% |
    | MoePermuteNopadFwd | `kimi-k2-decode-bfloat16` | 0.0088 | – | – | – |
    | MoePermuteNopadFwd | `kimi-k2-small-bfloat16` | 0.0095 | – | – | – |
    | MoePermuteNopadFwd | `kimi-k2-medium-bfloat16` | 0.0394 | – | – | – |
    | MoePermuteNopadFwd | `kimi-k2-prefill-bfloat16` | 0.5554 | – | – | – |
    | MoePermuteNopadFwd | `deepseek-v3-decode-bfloat16` | 0.0072 | – | – | – |
    | MoePermuteNopadFwd | `deepseek-v3-small-bfloat16` | 0.0082 | – | – | – |
    | MoePermuteNopadFwd | `deepseek-v3-medium-bfloat16` | 0.0373 | – | – | – |
    | MoePermuteNopadFwd | `deepseek-v3-prefill-bfloat16` | 0.5449 | – | – | – |
    | MoePermuteNopadFwd | `qwen3-235b-decode-bfloat16` | 0.0056 | – | – | – |
    | MoePermuteNopadFwd | `qwen3-235b-small-bfloat16` | 0.0069 | – | – | – |
    | MoePermuteNopadFwd | `qwen3-235b-medium-bfloat16` | 0.0349 | – | – | – |
    | MoePermuteNopadFwd | `qwen3-235b-prefill-bfloat16` | 0.4748 | – | – | – |
    | MoePermuteNopadFwd | `qwen3-30b-decode-bfloat16` | 0.0056 | – | – | – |
    | MoePermuteNopadFwd | `qwen3-30b-small-bfloat16` | 0.0064 | – | – | – |
    | MoePermuteNopadFwd | `qwen3-30b-medium-bfloat16` | 0.0203 | – | – | – |
    | MoePermuteNopadFwd | `qwen3-30b-prefill-bfloat16` | 0.2273 | – | – | – |
    | moe_unpermute | `large-hidden-decode-bfloat16` | 0.0074 | 0.0 | 1 | 0% |
    | moe_unpermute | `large-hidden-small-bfloat16` | 0.0083 | 0.4 | 1 | 10% |
    | moe_unpermute | `large-hidden-medium-bfloat16` | 0.0294 | 2.0 | 1 | 47% |
    | moe_unpermute | `large-hidden-prefill-bfloat16` | 0.189 | 2.5 | 1 | 58% |
    | moe_unpermute | `small-hidden-decode-bfloat16` | 0.0062 | 0.0 | 1 | 0% |
    | moe_unpermute | `small-hidden-small-bfloat16` | 0.007 | 0.2 | 1 | 5% |
    | moe_unpermute | `small-hidden-medium-bfloat16` | 0.0121 | 2.1 | 1 | 49% |
    | moe_unpermute | `small-hidden-prefill-bfloat16` | 0.0847 | 2.4 | 1 | 56% |

## Linear Algebra (GEMM)  <small>(2 ops)</small>

| Op | Tests | Configs | TFLOPS | % roof | vs baseline | Status |
| --- | :---: | ---: | ---: | ---: | --- | :---: |
| [Gemm](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/gemm.py) | ✅ | 4 | 2.3 | 44% | 🟢 1.02× torch-cublas | 🟢 |
| [grouped_gemm_3wg_baselines](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+grouped_gemm_3wg_baselines&type=code) | – | 8 | 642.1 | 65% | 🟢 0.99× deepgemm | 🟢 |

??? note "Per-config detail"

    | Op | Config | Latency (ms) | TFLOPS | AI | % roof |
    | --- | --- | ---: | ---: | ---: | ---: |
    | Gemm | `square-fp16` | 0.0092 | 233.5 | 343 | 24% |
    | Gemm | `wide-alt-bf16` | 0.1159 | 2.3 | 1 | 48% |
    | Gemm | `thin-n-fp16` | 0.1211 | 1.9 | 1 | 40% |
    | Gemm | `thin-n-alt-bf16` | 0.1144 | 2.3 | 1 | 48% |
    | grouped_gemm_3wg_baselines | `GLM-5-744B-up-T=32768` | 19.7067 | 669.5 | 720 | 68% |
    | grouped_gemm_3wg_baselines | `GLM-5-744B-up-T=65536` | 39.6891 | 664.9 | 1108 | 67% |
    | grouped_gemm_3wg_baselines | `GLM-5-744B-up-T=131072` | 155.1126 | 340.2 | 1547 | 34% |
    | grouped_gemm_3wg_baselines | `qwen3.5-397B-up-T52429` | 13.7648 | 639.0 | 586 | 65% |
    | grouped_gemm_3wg_baselines | `GLM-5-744B-down-T=32768` | 10.6603 | 618.9 | 613 | 63% |
    | grouped_gemm_3wg_baselines | `GLM-5-744B-down-T=65536` | 20.4534 | 645.1 | 884 | 65% |
    | grouped_gemm_3wg_baselines | `GLM-5-744B-down-T=131072` | 40.7104 | 648.2 | 1118 | 66% |
    | grouped_gemm_3wg_baselines | `qwen3.5-397B-down-T52429` | 7.1212 | 617.6 | 454 | 62% |

## Reduction  <small>(23 ops)</small>

| Op | Tests | Configs | TFLOPS | % roof | vs baseline | Status |
| --- | :---: | ---: | ---: | ---: | --- | :---: |
| [maximum](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+maximum&type=code) | – | 3 | 0.6 | 70% | 🟢 70% roof | 🟢 |
| [minimum](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+minimum&type=code) | – | 3 | 0.6 | 71% | 🟢 71% roof | 🟢 |
| [AllFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/reduction/all_op.py) | ✅ | 3 | 0.1 | 2% | 🔴 2% roof | 🔴 |
| [AmaxFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/reduction/reduce.py) | ✅ | 4 | 0.6 | 27% | 🔴 27% roof | 🔴 |
| [AminFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/reduction/reduce.py) | ✅ | 3 | 0.9 | 38% | 🔴 38% roof | 🔴 |
| [AnyFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/reduction/any_op.py) | ✅ | 4 | 0.1 | 3% | 🔴 3% roof | 🔴 |
| [CountNonzeroFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/reduction/count_nonzero.py) | ✅ | 4 | 1.1 | 22% | 🔴 22% roof | 🔴 |
| [CumprodFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/reduction/cumprod.py) | ✅ | 4 | 0.1 | 7% | 🔴 7% roof | 🔴 |
| [CumsumFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/reduction/cumsum.py) | ✅ | 5 | 0.0 | 4% | 🔴 4% roof | 🔴 |
| [InfNormFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/reduction/inf_norm.py) | ✅ | 4 | 0.3 | 7% | 🔴 7% roof | 🔴 |
| [L1NormFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/reduction/l1_norm.py) | ✅ | 4 | 1.3 | 27% | 🔴 27% roof | 🔴 |
| [L2NormFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/reduction/l2_norm.py) | ✅ | 5 | 0.7 | 15% | 🔴 15% roof | 🔴 |
| [LogSoftmaxFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/reduction/log_softmax.py) | ✅ | 7 | 1.6 | 30% | 🔴 30% roof | 🔴 |
| [LogSumExpFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/reduction/logsumexp.py) | ✅ | 11 | 0.4 | 5% | 🔴 5% roof | 🔴 |
| [MeanFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/reduction/reduce.py) | ✅ | 5 | 0.4 | 15% | 🔴 15% roof | 🔴 |
| [ProdFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/reduction/reduce.py) | ✅ | 3 | 0.4 | 18% | 🔴 18% roof | 🔴 |
| [SoftmaxFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/reduction/softmax.py) | ✅ | 7 | 1.6 | 30% | 🔴 30% roof | 🔴 |
| [StdFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/reduction/reduce.py) | ✅ | 3 | 4.0 | 34% | 🔴 34% roof | 🔴 |
| [SumFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/reduction/reduce.py) | ✅ | 8 | 0.3 | 12% | 🔴 12% roof | 🔴 |
| [VarFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/reduction/reduce.py) | ✅ | 3 | 4.0 | 34% | 🔴 34% roof | 🔴 |
| [VarMeanFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/reduction/reduce.py) | ✅ | 3 | 3.9 | 33% | 🔴 33% roof | 🔴 |
| [ArgmaxFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/reduction/argmax.py) | ✅ | 4 | – | – | — | — |
| [ArgminFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/reduction/argmin.py) | ✅ | 6 | – | – | — | — |

??? note "Per-config detail"

    | Op | Config | Latency (ms) | TFLOPS | AI | % roof |
    | --- | --- | ---: | ---: | ---: | ---: |
    | AllFwd | `mask-validation-4k-bool` | 0.0108 | 0.0 | 1 | 0% |
    | AllFwd | `mask-validation-32k-bool` | 0.0114 | 0.1 | 1 | 2% |
    | AllFwd | `all-7B-dim02-keepdim` | 0.0177 | 0.1 | 0 | 5% |
    | AmaxFwd | `hidden-state-reduce-float16` | 0.009 | 0.9 | 0 | 39% |
    | AmaxFwd | `hidden-state-reduce-bfloat16` | 0.0091 | 0.9 | 1 | 38% |
    | AmaxFwd | `long-seq-reduce-bfloat16` | 0.0058 | 0.4 | 0 | 15% |
    | AmaxFwd | `amax-7B-dim02-nokeepdim` | 0.0126 | 0.2 | 1 | 7% |
    | AminFwd | `hidden-state-reduce-float16` | 0.009 | 0.9 | 0 | 39% |
    | AminFwd | `hidden-state-reduce-bfloat16` | 0.0092 | 0.9 | 1 | 38% |
    | AminFwd | `long-seq-reduce-bfloat16` | 0.0058 | 0.4 | 0 | 15% |
    | AnyFwd | `mask-validation-4k-bool` | 0.0103 | 0.0 | 1 | 0% |
    | AnyFwd | `mask-validation-32k-bool` | 0.0112 | 0.1 | 1 | 2% |
    | AnyFwd | `any-7B-dim02-nokeepdim` | 0.0177 | 0.1 | 0 | 5% |
    | AnyFwd | `any-7B-dim01-keepdim` | 0.0245 | 0.1 | 1 | 4% |
    | ArgmaxFwd | `hidden-state-argmax-float16` | 4.0386 | – | – | – |
    | ArgmaxFwd | `hidden-state-argmax-bfloat16` | 4.0228 | – | – | – |
    | ArgmaxFwd | `argmax-7B-dim0-nokeepdim` | 2.9788 | – | – | – |
    | ArgmaxFwd | `argmax-7B-dim2-nokeepdim` | 2.2569 | – | – | – |
    | ArgminFwd | `hidden-state-argmin-float16` | 4.0215 | – | – | – |
    | ArgminFwd | `hidden-state-argmin-bfloat16` | 4.0119 | – | – | – |
    | ArgminFwd | `argmin-7B-dim0-keepdim-bf16` | 34.8379 | – | – | – |
    | ArgminFwd | `argmin-7B-dim1-nokeepdim` | 0.923 | – | – | – |
    | ArgminFwd | `argmin-7B-dim1-keepdim-bf16` | 0.9236 | – | – | – |
    | ArgminFwd | `argmin-7B-dim2-keepdim` | 2.2626 | – | – | – |
    | CountNonzeroFwd | `sparsity-hidden-float16` | 0.0089 | 1.9 | 1 | 39% |
    | CountNonzeroFwd | `sparsity-hidden-bfloat16` | 0.0097 | 1.7 | 1 | 36% |
    | CountNonzeroFwd | `sparsity-seq-float16` | 0.005 | 0.4 | 1 | 9% |
    | CountNonzeroFwd | `cnt_nz-7B-dim01-i32` | 0.0304 | 0.1 | 0 | 6% |
    | CumprodFwd | `hidden-state-scan-float16` | 0.1046 | 0.1 | 0 | 7% |
    | CumprodFwd | `hidden-state-scan-bfloat16` | 0.1047 | 0.1 | 0 | 7% |
    | CumprodFwd | `long-seq-scan-bfloat16` | 1.1527 | – | – | – |
    | CumprodFwd | `cumprod-7B-longctx-3D` | 0.0987 | 0.0 | 0 | 4% |
    | CumsumFwd | `hidden-state-scan-float16` | 0.1046 | 0.1 | 0 | 7% |
    | CumsumFwd | `hidden-state-scan-bfloat16` | 0.1048 | 0.1 | 0 | 7% |
    | CumsumFwd | `long-seq-scan-bfloat16` | 1.1528 | – | – | – |
    | CumsumFwd | `cumsum-7B-3D` | 0.0968 | 0.0 | 0 | 2% |
    | CumsumFwd | `cumsum-7B-3D-bf16` | 0.0856 | 0.0 | 0 | 2% |
    | InfNormFwd | `inf-7B-dim02-nokeepdim-bf16` | 0.0297 | 0.1 | 1 | 3% |
    | InfNormFwd | `hidden-state-inf-float16` | 0.0345 | 0.5 | 1 | 10% |
    | InfNormFwd | `hidden-state-inf-bfloat16` | 0.0347 | 0.5 | 1 | 10% |
    | InfNormFwd | `long-seq-inf-bfloat16` | 0.0216 | 0.2 | 1 | 4% |
    | L1NormFwd | `l1-7B-dim01-nokeepdim` | 0.021 | 0.2 | 1 | 4% |
    | L1NormFwd | `hidden-state-l1-float16` | 0.0091 | 1.8 | 1 | 38% |
    | L1NormFwd | `hidden-state-l1-bfloat16` | 0.009 | 1.9 | 1 | 39% |
    | L1NormFwd | `long-seq-l1-bfloat16` | 0.0057 | 0.7 | 1 | 15% |
    | L2NormFwd | `l2-7B-dim02-nokeepdim` | 0.015 | 0.3 | 1 | 6% |
    | L2NormFwd | `l2-7B-dim02-keepdim` | 0.0149 | 0.3 | 1 | 6% |
    | L2NormFwd | `hidden-state-l2-float16` | 0.0091 | 1.8 | 1 | 38% |
    | L2NormFwd | `hidden-state-l2-bfloat16` | 0.009 | 1.9 | 1 | 39% |
    | L2NormFwd | `long-seq-l2-bfloat16` | 0.0057 | 0.7 | 1 | 15% |
    | LogSoftmaxFwd | `attn-weights-4k-float16` | 0.0118 | 1.8 | 1 | 30% |
    | LogSoftmaxFwd | `attn-weights-4k-bfloat16` | 0.0093 | 2.3 | 1 | 38% |
    | LogSoftmaxFwd | `attn-weights-4k-fp32-float32` | 0.0134 | 1.6 | 1 | 52% |
    | LogSoftmaxFwd | `attn-weights-32k-bfloat16` | 0.0601 | 2.8 | 1 | 46% |
    | LogSoftmaxFwd | `lm-head-logits-float16` | 0.0267 | 0.1 | 1 | 1% |
    | LogSoftmaxFwd | `lm-head-logits-bfloat16` | 0.0274 | 0.1 | 1 | 1% |
    | LogSoftmaxFwd | `lm-head-logits-fp32-float32` | 0.0351 | 0.1 | 1 | 2% |
    | LogSumExpFwd | `lse-7B-dim02-nokeepdim` | 0.0191 | 0.4 | 2 | 5% |
    | LogSumExpFwd | `lse-7B-dim02-keepdim` | 0.0191 | 0.4 | 2 | 5% |
    | LogSumExpFwd | `lse-7B-dim01-nokeepdim` | 0.0225 | 0.4 | 2 | 4% |
    | LogSumExpFwd | `lse-7B-dim01-keepdim-bf16` | 0.0226 | 0.4 | 2 | 4% |
    | LogSumExpFwd | `lse-7B-longctx-dim02` | 0.0265 | 0.6 | 2 | 7% |
    | LogSumExpFwd | `lse-7B-longctx-dim02-keepdim-bf16` | 0.0267 | 0.6 | 2 | 6% |
    | LogSumExpFwd | `attn-weights-4k-float16` | 0.0074 | 2.3 | 2 | 24% |
    | LogSumExpFwd | `attn-weights-4k-bfloat16` | 0.0081 | 2.1 | 2 | 22% |
    | LogSumExpFwd | `attn-weights-32k-bfloat16` | 0.0384 | 3.5 | 2 | 36% |
    | LogSumExpFwd | `lm-head-logits-float16` | 0.0146 | 0.1 | 2 | 1% |
    | LogSumExpFwd | `lm-head-logits-bfloat16` | 0.0157 | 0.1 | 2 | 1% |
    | MeanFwd | `hidden-state-reduce-float16` | 0.009 | 0.9 | 0 | 39% |
    | MeanFwd | `hidden-state-reduce-bfloat16` | 0.0091 | 0.9 | 0 | 38% |
    | MeanFwd | `long-seq-reduce-bfloat16` | 0.0058 | 0.4 | 0 | 15% |
    | MeanFwd | `mean-7B-dim01-nokeepdim` | 0.0211 | 0.1 | 0 | 4% |
    | MeanFwd | `mean-7B-dim01-keepdim-bf16` | 0.021 | 0.1 | 0 | 4% |
    | ProdFwd | `hidden-state-reduce-float16` | 0.0184 | 0.5 | 1 | 19% |
    | ProdFwd | `hidden-state-reduce-bfloat16` | 0.019 | 0.4 | 0 | 18% |
    | ProdFwd | `long-seq-reduce-bfloat16` | 0.0176 | 0.1 | 0 | 5% |
    | SoftmaxFwd | `attn-weights-4k-float16` | 0.0118 | 1.8 | 1 | 30% |
    | SoftmaxFwd | `attn-weights-4k-bfloat16` | 0.0113 | 1.9 | 1 | 31% |
    | SoftmaxFwd | `attn-weights-4k-fp32-float32` | 0.0134 | 1.6 | 1 | 52% |
    | SoftmaxFwd | `attn-weights-32k-bfloat16` | 0.0718 | 2.3 | 1 | 39% |
    | SoftmaxFwd | `lm-head-logits-float16` | 0.0302 | 0.1 | 1 | 1% |
    | SoftmaxFwd | `lm-head-logits-bfloat16` | 0.0323 | 0.1 | 1 | 1% |
    | SoftmaxFwd | `lm-head-logits-fp32-float32` | 0.0396 | 0.1 | 1 | 2% |
    | StdFwd | `hidden-state-std-float16` | 0.0103 | 4.0 | 2 | 34% |
    | StdFwd | `hidden-state-std-bfloat16` | 0.0103 | 4.1 | 3 | 34% |
    | StdFwd | `long-seq-std-bfloat16` | 0.0077 | 1.4 | 3 | 11% |
    | SumFwd | `hidden-state-reduce-float16` | 0.009 | 0.9 | 0 | 39% |
    | SumFwd | `hidden-state-reduce-bfloat16` | 0.0091 | 0.9 | 0 | 38% |
    | SumFwd | `long-seq-reduce-bfloat16` | 0.0058 | 0.4 | 0 | 15% |
    | SumFwd | `hidden-state-reduce-dim0-bfloat16` | 0.0687 | 0.1 | 0 | 5% |
    | SumFwd | `hidden-state-reduce-keepdim-bfloat16` | 0.0091 | 0.9 | 0 | 39% |
    | SumFwd | `sum-7B-dim02-nokeepdim` | 0.0127 | 0.2 | 1 | 7% |
    | SumFwd | `sum-7B-dim02-keepdim` | 0.0127 | 0.2 | 1 | 7% |
    | SumFwd | `sum-7B-longctx-dim02` | 0.0195 | 0.2 | 0 | 9% |
    | VarFwd | `hidden-state-var-float16` | 0.0103 | 4.1 | 2 | 34% |
    | VarFwd | `hidden-state-var-bfloat16` | 0.0104 | 4.0 | 3 | 34% |
    | VarFwd | `long-seq-var-bfloat16` | 0.0077 | 1.4 | 2 | 11% |
    | VarMeanFwd | `hidden-state-var-mean-float16` | 0.0105 | 4.0 | 2 | 33% |
    | VarMeanFwd | `hidden-state-var-mean-bfloat16` | 0.0107 | 3.9 | 3 | 33% |
    | VarMeanFwd | `long-seq-var-mean-bfloat16` | 0.0077 | 1.4 | 3 | 11% |
    | maximum | `maximum-shape17-dtype17-output_dtype17-MaximumFwdOp-maximum-_randn_pair` | 0.0092 | 0.5 | 0 | 57% |
    | maximum | `maximum-shape18-dtype18-output_dtype18-MaximumFwdOp-maximum-_randn_pair` | 0.0186 | 0.6 | 0 | 70% |
    | maximum | `maximum-shape19-dtype19-output_dtype19-MaximumFwdOp-maximum-_randn_pair` | 0.0196 | 0.6 | 0 | 72% |
    | minimum | `minimum-shape20-dtype20-output_dtype20-MinimumFwdOp-minimum-_randn_pair` | 0.0092 | 0.5 | 0 | 57% |
    | minimum | `minimum-shape21-dtype21-output_dtype21-MinimumFwdOp-minimum-_randn_pair` | 0.0185 | 0.6 | 0 | 71% |
    | minimum | `minimum-shape22-dtype22-output_dtype22-MinimumFwdOp-minimum-_randn_pair` | 0.0196 | 0.6 | 0 | 72% |

## Elementwise  <small>(102 ops)</small>

| Op | Tests | Configs | TFLOPS | % roof | vs baseline | Status |
| --- | :---: | ---: | ---: | ---: | --- | :---: |
| [AbsFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/math_unary.py) | ✅ | 5 | 0.8 | 75% | 🟢 75% roof | 🟢 |
| [BitwiseAndFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/bitwise.py) | ✅ | 6 | 0.3 | 75% | 🟢 75% roof | 🟢 |
| [BitwiseOrFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/bitwise.py) | ✅ | 6 | 0.3 | 74% | 🟢 74% roof | 🟢 |
| [BitwiseXorFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/bitwise.py) | ✅ | 6 | 0.3 | 75% | 🟢 75% roof | 🟢 |
| [CeilFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/math_unary.py) | ✅ | 5 | 0.8 | 74% | 🟢 74% roof | 🟢 |
| [ClampFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/clamp.py) | – | 5 | 0.4 | 73% | 🟢 73% roof | 🟢 |
| [ClampMaxFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/clamp.py) | – | 5 | 0.5 | 72% | 🟢 72% roof | 🟢 |
| [ClampMinFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/clamp.py) | – | 5 | 0.5 | 70% | 🟢 70% roof | 🟢 |
| [ClampScalarFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/clamp.py) | – | 5 | 0.6 | 74% | 🟢 74% roof | 🟢 |
| [ExpFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/math_unary.py) | ✅ | 5 | 0.8 | 73% | 🟢 73% roof | 🟢 |
| [Expm1Fwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/math_unary.py) | ✅ | 5 | 1.6 | 74% | 🟢 74% roof | 🟢 |
| [FloorFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/math_unary.py) | ✅ | 5 | 0.8 | 75% | 🟢 75% roof | 🟢 |
| [MaskedFillFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/masked_fill.py) | – | 10 | 0.6 | 75% | 🟢 75% roof | 🟢 |
| [NanToNumFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/nan_to_num.py) | – | 5 | 3.4 | 71% | 🟢 71% roof | 🟢 |
| [NegFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/math_unary.py) | ✅ | 5 | 0.8 | 75% | 🟢 75% roof | 🟢 |
| [RoundFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/math_unary.py) | ✅ | 5 | 0.8 | 74% | 🟢 74% roof | 🟢 |
| [RsqrtFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/math_unary.py) | ✅ | 5 | 0.8 | 75% | 🟢 75% roof | 🟢 |
| [SignFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/math_unary.py) | ✅ | 5 | 1.5 | 70% | 🟢 70% roof | 🟢 |
| [SqrtFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/math_unary.py) | ✅ | 5 | 0.8 | 71% | 🟢 71% roof | 🟢 |
| [TruncFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/math_unary.py) | ✅ | 5 | 0.8 | 74% | 🟢 74% roof | 🟢 |
| [WhereFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/where.py) | – | 14 | 0.4 | 72% | 🟢 72% roof | 🟢 |
| [bitwise_and](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+bitwise_and&type=code) | – | 2 | 0.3 | 73% | 🟢 73% roof | 🟢 |
| [div](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+div&type=code) | – | 3 | 0.6 | 71% | 🟢 71% roof | 🟢 |
| [mul](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+mul&type=code) | – | 3 | 0.6 | 71% | 🟢 71% roof | 🟢 |
| [sub](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+sub&type=code) | – | 3 | 0.6 | 72% | 🟢 72% roof | 🟢 |
| [AddFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/arithmetic.py) | ✅ | 9 | 0.9 | 69% | 🟡 69% roof | 🟡 |
| [CosFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/math_unary.py) | ✅ | 5 | 0.5 | 51% | 🟡 51% roof | 🟡 |
| [DivFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/arithmetic.py) | ✅ | 6 | 0.5 | 68% | 🟡 68% roof | 🟡 |
| [Dropout](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/dropout.py) | – | 9 | 0.6 | 68% | 🟡 68% roof | 🟡 |
| [EluFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/activations.py) | ✅ | 5 | 2.7 | 56% | 🟡 56% roof | 🟡 |
| [ErfFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/math_unary.py) | ✅ | 5 | 0.5 | 47% | 🟡 47% roof | 🟡 |
| [FloorDivideFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/arithmetic.py) | ✅ | 6 | 1.1 | 67% | 🟡 67% roof | 🟡 |
| [HardswishFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/activations.py) | ✅ | 5 | 2.6 | 53% | 🟡 53% roof | 🟡 |
| [HardtanhFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/activations.py) | ✅ | 5 | 0.8 | 65% | 🟡 65% roof | 🟡 |
| [LeakyReluFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/activations.py) | ✅ | 5 | 1.6 | 65% | 🟡 65% roof | 🟡 |
| [LerpFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/arithmetic.py) | ✅ | 6 | 1.6 | 68% | 🟡 68% roof | 🟡 |
| [LerpTensorFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/arithmetic.py) | – | 10 | 1.1 | 67% | 🟡 67% roof | 🟡 |
| [Log1pFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/math_unary.py) | ✅ | 5 | 1.0 | 47% | 🟡 47% roof | 🟡 |
| [LogFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/math_unary.py) | ✅ | 5 | 0.5 | 48% | 🟡 48% roof | 🟡 |
| [MaskedFillScalarFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/masked_fill.py) | – | 14 | 0.5 | 70% | 🟡 70% roof | 🟡 |
| [MaximumFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/arithmetic.py) | ✅ | 6 | 0.4 | 67% | 🟡 67% roof | 🟡 |
| [MinimumFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/arithmetic.py) | ✅ | 6 | 0.4 | 67% | 🟡 67% roof | 🟡 |
| [MulFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/arithmetic.py) | ✅ | 6 | 0.6 | 68% | 🟡 68% roof | 🟡 |
| [PreluFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/prelu.py) | – | 16 | 0.7 | 67% | 🟡 67% roof | 🟡 |
| [ReciprocalFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/math_unary.py) | ✅ | 5 | 0.7 | 69% | 🟡 69% roof | 🟡 |
| [ReluFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/activations.py) | ✅ | 7 | 0.7 | 65% | 🟡 65% roof | 🟡 |
| [RemainderFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/arithmetic.py) | ✅ | 6 | 2.1 | 67% | 🟡 67% roof | 🟡 |
| [SeluFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/activations.py) | ✅ | 5 | 3.4 | 56% | 🟡 56% roof | 🟡 |
| [SigmoidFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/activations.py) | ✅ | 5 | 1.7 | 48% | 🟡 48% roof | 🟡 |
| [SinFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/math_unary.py) | ✅ | 5 | 0.6 | 53% | 🟡 53% roof | 🟡 |
| [SubFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/arithmetic.py) | ✅ | 6 | 1.1 | 68% | 🟡 68% roof | 🟡 |
| [TanhFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/activations.py) | ✅ | 5 | 0.5 | 64% | 🟡 64% roof | 🟡 |
| [alibi](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+alibi&type=code) | – | 9 | 1.3 | 66% | 🟡 66% roof | 🟡 |
| [bitwise_or](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+bitwise_or&type=code) | – | 1 | 0.3 | 67% | 🟡 67% roof | 🟡 |
| [bitwise_xor](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+bitwise_xor&type=code) | – | 1 | 0.3 | 67% | 🟡 67% roof | 🟡 |
| [clamp](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+clamp&type=code) | – | 9 | 0.7 | 68% | 🟡 68% roof | 🟡 |
| [div_bcast](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+div_bcast&type=code) | – | 3 | 0.7 | 56% | 🟡 56% roof | 🟡 |
| [elu](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+elu&type=code) | – | 9 | 0.6 | 60% | 🟡 60% roof | 🟡 |
| [floor_divide](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+floor_divide&type=code) | – | 2 | 0.5 | 64% | 🟡 64% roof | 🟡 |
| [gelu_and_mul](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+gelu_and_mul&type=code) | – | 3 | 0.9 | 54% | 🟡 54% roof | 🟡 |
| [gelu_and_mul_strategy](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+gelu_and_mul_strategy&type=code) | – | 18 | 0.4 | 46% | 🟡 46% roof | 🟡 |
| [gelu_tanh_and_mul](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+gelu_tanh_and_mul&type=code) | – | 3 | 1.0 | 63% | 🟡 63% roof | 🟡 |
| [gelu_tanh_and_mul_strategy](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+gelu_tanh_and_mul_strategy&type=code) | – | 18 | 0.5 | 53% | 🟡 53% roof | 🟡 |
| [hardtanh](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+hardtanh&type=code) | – | 9 | 0.7 | 68% | 🟡 68% roof | 🟡 |
| [leaky_relu](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+leaky_relu&type=code) | – | 9 | 0.7 | 69% | 🟡 69% roof | 🟡 |
| [lerp](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+lerp&type=code) | – | 2 | 0.5 | 65% | 🟡 65% roof | 🟡 |
| [mul_bcast](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+mul_bcast&type=code) | – | 3 | 0.7 | 62% | 🟡 62% roof | 🟡 |
| [nan_to_num](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+nan_to_num&type=code) | – | 9 | 0.6 | 66% | 🟡 66% roof | 🟡 |
| [r4_where](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+r4_where&type=code) | – | 3 | 0.2 | 54% | 🟡 54% roof | 🟡 |
| [remainder](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+remainder&type=code) | – | 2 | 0.5 | 64% | 🟡 64% roof | 🟡 |
| [silu_and_mul_strategy](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+silu_and_mul_strategy&type=code) | – | 18 | 0.5 | 52% | 🟡 52% roof | 🟡 |
| [sub_bcast](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+sub_bcast&type=code) | – | 3 | 0.7 | 61% | 🟡 61% roof | 🟡 |
| [BitwiseNotFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/bitwise.py) | ✅ | 3 | 0.2 | 36% | 🔴 36% roof | 🔴 |
| [EqFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/comparison.py) | ✅ | 6 | 0.2 | 24% | 🔴 24% roof | 🔴 |
| [GeFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/comparison.py) | ✅ | 6 | 0.2 | 24% | 🔴 24% roof | 🔴 |
| [GeluFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/activations.py) | ✅ | 3 | 1.9 | 32% | 🔴 32% roof | 🔴 |
| [GtFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/comparison.py) | ✅ | 6 | 0.2 | 23% | 🔴 23% roof | 🔴 |
| [HardsigmoidFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/activations.py) | ✅ | 5 | 0.0 | 1% | 🔴 1% roof | 🔴 |
| [IsfiniteFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/predicates.py) | ✅ | 5 | 0.2 | 15% | 🔴 15% roof | 🔴 |
| [IsinfFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/predicates.py) | ✅ | 5 | 0.2 | 16% | 🔴 16% roof | 🔴 |
| [IsnanFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/predicates.py) | ✅ | 5 | 0.2 | 16% | 🔴 16% roof | 🔴 |
| [LeFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/comparison.py) | ✅ | 6 | 0.2 | 24% | 🔴 24% roof | 🔴 |
| [LogicalAndFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/logical.py) | ✅ | 8 | 0.7 | 20% | 🔴 20% roof | 🔴 |
| [LogicalNotFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/logical.py) | ✅ | 4 | 0.3 | 14% | 🔴 14% roof | 🔴 |
| [LogicalOrFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/logical.py) | ✅ | 8 | 0.7 | 20% | 🔴 20% roof | 🔴 |
| [LtFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/comparison.py) | ✅ | 6 | 0.2 | 24% | 🔴 24% roof | 🔴 |
| [MishFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/activations.py) | ✅ | 5 | 1.2 | 25% | 🔴 25% roof | 🔴 |
| [NeFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/comparison.py) | ✅ | 6 | 0.2 | 24% | 🔴 24% roof | 🔴 |
| [PowFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/arithmetic.py) | ✅ | 6 | 0.6 | 28% | 🔴 28% roof | 🔴 |
| [SiluFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/activations.py) | ✅ | 4 | 2.2 | 37% | 🔴 37% roof | 🔴 |
| [SoftplusFwd](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/elementwise/activations.py) | ✅ | 5 | 2.1 | 35% | 🔴 35% roof | 🔴 |
| [cmp_eq](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+cmp_eq&type=code) | – | 2 | 0.2 | 23% | 🔴 23% roof | 🔴 |
| [cmp_ge](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+cmp_ge&type=code) | – | 1 | 0.2 | 21% | 🔴 21% roof | 🔴 |
| [cmp_gt](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+cmp_gt&type=code) | – | 1 | 0.2 | 21% | 🔴 21% roof | 🔴 |
| [cmp_le](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+cmp_le&type=code) | – | 1 | 0.2 | 21% | 🔴 21% roof | 🔴 |
| [cmp_lt](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+cmp_lt&type=code) | – | 1 | 0.2 | 21% | 🔴 21% roof | 🔴 |
| [cmp_ne](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+cmp_ne&type=code) | – | 1 | 0.2 | 21% | 🔴 21% roof | 🔴 |
| [logical_and](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+logical_and&type=code) | – | 2 | 0.1 | 15% | 🔴 15% roof | 🔴 |
| [logical_or](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+logical_or&type=code) | – | 2 | 0.2 | 23% | 🔴 23% roof | 🔴 |
| [pow](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+pow&type=code) | – | 2 | 0.2 | 27% | 🔴 27% roof | 🔴 |
| [sinusoidal](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+sinusoidal&type=code) | – | 9 | 0.1 | 5% | 🔴 5% roof | 🔴 |
| [softplus](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+softplus&type=code) | – | 9 | 0.4 | 37% | 🔴 37% roof | 🔴 |

??? note "Per-config detail"

    | Op | Config | Latency (ms) | TFLOPS | AI | % roof |
    | --- | --- | ---: | ---: | ---: | ---: |
    | AbsFwd | `elementwise-16M-float16` | 0.0187 | 0.9 | 0 | 75% |
    | AbsFwd | `elementwise-16M-bfloat16` | 0.0186 | 0.9 | 0 | 75% |
    | AbsFwd | `elementwise-16M-float32` | 0.0344 | 0.5 | 0 | 81% |
    | AbsFwd | `elementwise-256M-float16` | 0.3336 | 0.8 | 0 | 67% |
    | AbsFwd | `elementwise-256M-bfloat16` | 0.3334 | 0.8 | 0 | 67% |
    | AddFwd | `throughput-fp16` | 0.0088 | 0.5 | 0 | 59% |
    | AddFwd | `throughput-bf16` | 0.0089 | 0.5 | 0 | 59% |
    | AddFwd | `baseline-fp32` | 0.0152 | 0.3 | 0 | 69% |
    | AddFwd | `hidden-state-prefill-float16` | 0.0153 | 1.1 | 0 | 69% |
    | AddFwd | `hidden-state-prefill-bfloat16` | 0.0153 | 1.1 | 0 | 69% |
    | AddFwd | `hidden-state-prefill-float32` | 0.0273 | 0.6 | 0 | 77% |
    | AddFwd | `cnn-feat-broadcast-float16` | 0.0172 | 1.5 | 0 | 62% |
    | AddFwd | `cnn-feat-broadcast-bfloat16` | 0.0173 | 1.5 | 0 | 62% |
    | AddFwd | `cnn-feat-broadcast-float32` | 0.0271 | 0.9 | 0 | 79% |
    | BitwiseAndFwd | `hidden-state-prefill-bool` | 0.0285 | 0.3 | 0 | 18% |
    | BitwiseAndFwd | `hidden-state-prefill-int32` | 0.0277 | 0.3 | 0 | 76% |
    | BitwiseAndFwd | `hidden-state-prefill-int64` | 0.0531 | 0.2 | 0 | 79% |
    | BitwiseAndFwd | `cnn-feat-broadcast-bool` | 0.0419 | 0.3 | 1 | 13% |
    | BitwiseAndFwd | `cnn-feat-broadcast-int32` | 0.0279 | 0.5 | 0 | 77% |
    | BitwiseAndFwd | `cnn-feat-broadcast-int64` | 0.0582 | 0.2 | 0 | 74% |
    | BitwiseNotFwd | `elementwise-16M-int32` | 0.0812 | 0.2 | 0 | 34% |
    | BitwiseNotFwd | `elementwise-16M-int64` | 0.0995 | 0.2 | 0 | 56% |
    | BitwiseNotFwd | `elementwise-256M-int32` | 1.2397 | 0.2 | 0 | 36% |
    | BitwiseOrFwd | `hidden-state-prefill-bool` | 0.0285 | 0.3 | 0 | 18% |
    | BitwiseOrFwd | `hidden-state-prefill-int32` | 0.0277 | 0.3 | 0 | 76% |
    | BitwiseOrFwd | `hidden-state-prefill-int64` | 0.0541 | 0.2 | 0 | 78% |
    | BitwiseOrFwd | `cnn-feat-broadcast-bool` | 0.0416 | 0.3 | 0 | 13% |
    | BitwiseOrFwd | `cnn-feat-broadcast-int32` | 0.0279 | 0.5 | 0 | 77% |
    | BitwiseOrFwd | `cnn-feat-broadcast-int64` | 0.0592 | 0.2 | 0 | 72% |
    | BitwiseXorFwd | `hidden-state-prefill-bool` | 0.0285 | 0.3 | 0 | 18% |
    | BitwiseXorFwd | `hidden-state-prefill-int32` | 0.0277 | 0.3 | 0 | 76% |
    | BitwiseXorFwd | `hidden-state-prefill-int64` | 0.051 | 0.2 | 0 | 82% |
    | BitwiseXorFwd | `cnn-feat-broadcast-bool` | 0.0419 | 0.3 | 1 | 13% |
    | BitwiseXorFwd | `cnn-feat-broadcast-int32` | 0.028 | 0.5 | 0 | 76% |
    | BitwiseXorFwd | `cnn-feat-broadcast-int64` | 0.0582 | 0.2 | 0 | 74% |
    | CeilFwd | `elementwise-16M-float16` | 0.0188 | 0.9 | 0 | 74% |
    | CeilFwd | `elementwise-16M-bfloat16` | 0.0188 | 0.9 | 0 | 75% |
    | CeilFwd | `elementwise-16M-float32` | 0.0344 | 0.5 | 0 | 81% |
    | CeilFwd | `elementwise-256M-float16` | 0.3354 | 0.8 | 0 | 67% |
    | CeilFwd | `elementwise-256M-bfloat16` | 0.3355 | 0.8 | 0 | 67% |
    | ClampFwd | `elementwise-16M-float16` | 0.0359 | 0.5 | 0 | 78% |
    | ClampFwd | `elementwise-16M-bfloat16` | 0.0359 | 0.5 | 0 | 78% |
    | ClampFwd | `elementwise-16M-float32` | 0.0769 | 0.2 | 0 | 73% |
    | ClampFwd | `elementwise-256M-float16` | 0.6918 | 0.4 | 0 | 65% |
    | ClampFwd | `elementwise-256M-bfloat16` | 0.6918 | 0.4 | 0 | 65% |
    | ClampMaxFwd | `elementwise-16M-float16` | 0.0276 | 0.6 | 0 | 76% |
    | ClampMaxFwd | `elementwise-16M-bfloat16` | 0.0276 | 0.6 | 0 | 76% |
    | ClampMaxFwd | `elementwise-16M-float32` | 0.0579 | 0.3 | 0 | 72% |
    | ClampMaxFwd | `elementwise-256M-float16` | 0.558 | 0.5 | 0 | 60% |
    | ClampMaxFwd | `elementwise-256M-bfloat16` | 0.5284 | 0.5 | 0 | 64% |
    | ClampMinFwd | `elementwise-16M-float16` | 0.0276 | 0.6 | 0 | 76% |
    | ClampMinFwd | `elementwise-16M-bfloat16` | 0.0277 | 0.6 | 0 | 76% |
    | ClampMinFwd | `elementwise-16M-float32` | 0.0599 | 0.3 | 0 | 70% |
    | ClampMinFwd | `elementwise-256M-float16` | 0.5285 | 0.5 | 0 | 64% |
    | ClampMinFwd | `elementwise-256M-bfloat16` | 0.5285 | 0.5 | 0 | 64% |
    | ClampScalarFwd | `elementwise-16M-float16` | 0.0189 | 0.9 | 0 | 74% |
    | ClampScalarFwd | `elementwise-16M-bfloat16` | 0.0188 | 0.9 | 0 | 74% |
    | ClampScalarFwd | `elementwise-16M-float32` | 0.0373 | 0.5 | 0 | 75% |
    | ClampScalarFwd | `elementwise-256M-float16` | 0.4556 | 0.6 | 0 | 49% |
    | ClampScalarFwd | `elementwise-256M-bfloat16` | 0.4561 | 0.6 | 0 | 49% |
    | CosFwd | `elementwise-16M-float16` | 0.0269 | 0.6 | 0 | 52% |
    | CosFwd | `elementwise-16M-bfloat16` | 0.0272 | 0.6 | 0 | 51% |
    | CosFwd | `elementwise-16M-float32` | 0.0365 | 0.5 | 0 | 76% |
    | CosFwd | `elementwise-256M-float16` | 0.5 | 0.5 | 0 | 45% |
    | CosFwd | `elementwise-256M-bfloat16` | 0.507 | 0.5 | 0 | 44% |
    | DivFwd | `hidden-state-prefill-float16` | 0.0154 | 0.5 | 0 | 68% |
    | DivFwd | `hidden-state-prefill-bfloat16` | 0.0154 | 0.5 | 0 | 68% |
    | DivFwd | `hidden-state-prefill-float32` | 0.0273 | 0.3 | 0 | 77% |
    | DivFwd | `cnn-feat-broadcast-float16` | 0.0193 | 0.7 | 0 | 55% |
    | DivFwd | `cnn-feat-broadcast-bfloat16` | 0.0179 | 0.7 | 0 | 60% |
    | DivFwd | `cnn-feat-broadcast-float32` | 0.0278 | 0.5 | 0 | 77% |
    | Dropout | `shape0-dtype0` | 0.0068 | 0.6 | 0 | 52% |
    | Dropout | `shape1-dtype1` | 0.0068 | 0.6 | 0 | 51% |
    | Dropout | `shape2-dtype2` | 0.0109 | 0.4 | 0 | 64% |
    | Dropout | `shape3-dtype3` | 0.0129 | 0.8 | 0 | 68% |
    | Dropout | `shape4-dtype4` | 0.013 | 0.8 | 0 | 67% |
    | Dropout | `shape5-dtype5` | 0.023 | 0.5 | 0 | 76% |
    | Dropout | `shape6-dtype6` | 0.0138 | 0.8 | 0 | 68% |
    | Dropout | `shape7-dtype7` | 0.0139 | 0.8 | 0 | 68% |
    | Dropout | `shape8-dtype8` | 0.0244 | 0.5 | 0 | 77% |
    | EluFwd | `dtype0-elu` | 0.0108 | 1.6 | 1 | 32% |
    | EluFwd | `mlp-hidden-float16` | 0.0125 | 2.7 | 1 | 56% |
    | EluFwd | `mlp-hidden-bfloat16` | 0.0125 | 2.7 | 1 | 56% |
    | EluFwd | `mlp-hidden-wide-float16` | 0.0222 | 3.0 | 1 | 63% |
    | EluFwd | `mlp-hidden-wide-bfloat16` | 0.0222 | 3.0 | 1 | 63% |
    | EqFwd | `hidden-state-prefill-float16` | 0.0369 | 0.2 | 0 | 24% |
    | EqFwd | `hidden-state-prefill-bfloat16` | 0.0367 | 0.2 | 0 | 24% |
    | EqFwd | `hidden-state-prefill-float32` | 0.0463 | 0.2 | 0 | 34% |
    | EqFwd | `cnn-feat-broadcast-float16` | 0.0504 | 0.2 | 0 | 16% |
    | EqFwd | `cnn-feat-broadcast-bfloat16` | 0.0504 | 0.2 | 0 | 16% |
    | EqFwd | `cnn-feat-broadcast-float32` | 0.0574 | 0.2 | 0 | 23% |
    | ErfFwd | `elementwise-16M-float16` | 0.0293 | 0.6 | 0 | 48% |
    | ErfFwd | `elementwise-16M-bfloat16` | 0.0296 | 0.6 | 0 | 47% |
    | ErfFwd | `elementwise-16M-float32` | 0.0366 | 0.5 | 0 | 76% |
    | ErfFwd | `elementwise-256M-float16` | 0.5708 | 0.5 | 0 | 39% |
    | ErfFwd | `elementwise-256M-bfloat16` | 0.5704 | 0.5 | 0 | 39% |
    | ExpFwd | `elementwise-16M-float16` | 0.0189 | 0.9 | 0 | 74% |
    | ExpFwd | `elementwise-16M-bfloat16` | 0.019 | 0.9 | 0 | 73% |
    | ExpFwd | `elementwise-16M-float32` | 0.0344 | 0.5 | 0 | 81% |
    | ExpFwd | `elementwise-256M-float16` | 0.3418 | 0.8 | 0 | 65% |
    | ExpFwd | `elementwise-256M-bfloat16` | 0.3451 | 0.8 | 0 | 65% |
    | Expm1Fwd | `elementwise-16M-float16` | 0.0189 | 1.8 | 0 | 74% |
    | Expm1Fwd | `elementwise-16M-bfloat16` | 0.019 | 1.8 | 0 | 74% |
    | Expm1Fwd | `elementwise-16M-float32` | 0.0344 | 1.0 | 0 | 81% |
    | Expm1Fwd | `elementwise-256M-float16` | 0.3414 | 1.6 | 0 | 65% |
    | Expm1Fwd | `elementwise-256M-bfloat16` | 0.3448 | 1.6 | 1 | 65% |
    | FloorDivideFwd | `hidden-state-prefill-float16` | 0.0156 | 1.1 | 0 | 67% |
    | FloorDivideFwd | `hidden-state-prefill-bfloat16` | 0.0156 | 1.1 | 0 | 67% |
    | FloorDivideFwd | `hidden-state-prefill-float32` | 0.0272 | 0.6 | 0 | 77% |
    | FloorDivideFwd | `cnn-feat-broadcast-float16` | 0.0204 | 1.3 | 1 | 52% |
    | FloorDivideFwd | `cnn-feat-broadcast-bfloat16` | 0.0205 | 1.2 | 0 | 52% |
    | FloorDivideFwd | `cnn-feat-broadcast-float32` | 0.028 | 0.9 | 0 | 77% |
    | FloorFwd | `elementwise-16M-float16` | 0.0187 | 0.9 | 0 | 75% |
    | FloorFwd | `elementwise-16M-bfloat16` | 0.0187 | 0.9 | 0 | 75% |
    | FloorFwd | `elementwise-16M-float32` | 0.0343 | 0.5 | 0 | 81% |
    | FloorFwd | `elementwise-256M-float16` | 0.3354 | 0.8 | 0 | 67% |
    | FloorFwd | `elementwise-256M-bfloat16` | 0.3355 | 0.8 | 0 | 67% |
    | GeFwd | `hidden-state-prefill-float16` | 0.0369 | 0.2 | 0 | 24% |
    | GeFwd | `hidden-state-prefill-bfloat16` | 0.0367 | 0.2 | 0 | 24% |
    | GeFwd | `hidden-state-prefill-float32` | 0.0463 | 0.2 | 0 | 34% |
    | GeFwd | `cnn-feat-broadcast-float16` | 0.0505 | 0.2 | 0 | 16% |
    | GeFwd | `cnn-feat-broadcast-bfloat16` | 0.0505 | 0.2 | 0 | 16% |
    | GeFwd | `cnn-feat-broadcast-float32` | 0.0575 | 0.2 | 0 | 23% |
    | GeluFwd | `llama-3.1-8b-ffn-prefill-float16` | 0.0719 | 2.0 | 1 | 34% |
    | GeluFwd | `llama-3.1-8b-ffn-prefill-bfloat16` | 0.0756 | 1.9 | 1 | 32% |
    | GeluFwd | `llama-3.1-8b-ffn-decode-bfloat16` | 0.0021 | 0.0 | 1 | 1% |
    | GtFwd | `hidden-state-prefill-float16` | 0.0367 | 0.2 | 0 | 24% |
    | GtFwd | `hidden-state-prefill-bfloat16` | 0.0367 | 0.2 | 0 | 24% |
    | GtFwd | `hidden-state-prefill-float32` | 0.0464 | 0.2 | 0 | 34% |
    | GtFwd | `cnn-feat-broadcast-float16` | 0.0503 | 0.3 | 0 | 16% |
    | GtFwd | `cnn-feat-broadcast-bfloat16` | 0.0504 | 0.3 | 0 | 16% |
    | GtFwd | `cnn-feat-broadcast-float32` | 0.0577 | 0.2 | 0 | 23% |
    | HardsigmoidFwd | `dtype0-hardsigmoid` | 0.0071 | 1.8 | 1 | 49% |
    | HardsigmoidFwd | `mbv3-se-gate-float16` | 0.0025 | 0.0 | 1 | 0% |
    | HardsigmoidFwd | `mbv3-se-gate-bfloat16` | 0.0027 | 0.0 | 1 | 0% |
    | HardsigmoidFwd | `mbv3-se-gate-deep-float16` | 0.0024 | 0.0 | 1 | 1% |
    | HardsigmoidFwd | `mbv3-se-gate-deep-bfloat16` | 0.0026 | 0.0 | 1 | 1% |
    | HardswishFwd | `dtype0-hardswish` | 0.0073 | 2.3 | 1 | 48% |
    | HardswishFwd | `mbv3-stage2-float16` | 0.0134 | 2.9 | 1 | 60% |
    | HardswishFwd | `mbv3-stage2-bfloat16` | 0.0137 | 2.8 | 1 | 59% |
    | HardswishFwd | `mbv3-stage3-float16` | 0.0094 | 2.6 | 1 | 53% |
    | HardswishFwd | `mbv3-stage3-bfloat16` | 0.0095 | 2.5 | 1 | 53% |
    | HardtanhFwd | `dtype0-hardtanh` | 0.0065 | 0.7 | 0 | 54% |
    | HardtanhFwd | `bounded-hidden-float16` | 0.0107 | 0.8 | 0 | 65% |
    | HardtanhFwd | `bounded-hidden-bfloat16` | 0.0107 | 0.8 | 0 | 65% |
    | HardtanhFwd | `bounded-conv-feat-float16` | 0.015 | 0.8 | 0 | 71% |
    | HardtanhFwd | `bounded-conv-feat-bfloat16` | 0.0151 | 0.8 | 0 | 71% |
    | IsfiniteFwd | `elementwise-16M-float16` | 0.0686 | 0.2 | 0 | 15% |
    | IsfiniteFwd | `elementwise-16M-bfloat16` | 0.0661 | 0.2 | 0 | 16% |
    | IsfiniteFwd | `elementwise-16M-float32` | 0.0687 | 0.2 | 0 | 25% |
    | IsfiniteFwd | `elementwise-256M-float16` | 1.0934 | 0.2 | 0 | 15% |
    | IsfiniteFwd | `elementwise-256M-bfloat16` | 1.0933 | 0.2 | 0 | 15% |
    | IsinfFwd | `elementwise-16M-float16` | 0.0658 | 0.3 | 0 | 16% |
    | IsinfFwd | `elementwise-16M-bfloat16` | 0.0652 | 0.3 | 0 | 16% |
    | IsinfFwd | `elementwise-16M-float32` | 0.0701 | 0.2 | 0 | 25% |
    | IsinfFwd | `elementwise-256M-float16` | 1.1068 | 0.2 | 0 | 15% |
    | IsinfFwd | `elementwise-256M-bfloat16` | 1.0995 | 0.2 | 0 | 15% |
    | IsnanFwd | `elementwise-16M-float16` | 0.0661 | 0.2 | 0 | 16% |
    | IsnanFwd | `elementwise-16M-bfloat16` | 0.0662 | 0.2 | 0 | 16% |
    | IsnanFwd | `elementwise-16M-float32` | 0.0675 | 0.2 | 0 | 26% |
    | IsnanFwd | `elementwise-256M-float16` | 1.0931 | 0.2 | 0 | 15% |
    | IsnanFwd | `elementwise-256M-bfloat16` | 1.0932 | 0.2 | 0 | 15% |
    | LeFwd | `hidden-state-prefill-float16` | 0.0367 | 0.2 | 0 | 24% |
    | LeFwd | `hidden-state-prefill-bfloat16` | 0.0369 | 0.2 | 0 | 24% |
    | LeFwd | `hidden-state-prefill-float32` | 0.0462 | 0.2 | 0 | 34% |
    | LeFwd | `cnn-feat-broadcast-float16` | 0.0506 | 0.2 | 0 | 16% |
    | LeFwd | `cnn-feat-broadcast-bfloat16` | 0.0504 | 0.2 | 0 | 16% |
    | LeFwd | `cnn-feat-broadcast-float32` | 0.0575 | 0.2 | 0 | 23% |
    | LeakyReluFwd | `dtype0-leaky_relu` | 0.009 | 0.9 | 0 | 39% |
    | LeakyReluFwd | `gan-feat-float16` | 0.0189 | 1.8 | 0 | 74% |
    | LeakyReluFwd | `gan-feat-bfloat16` | 0.0189 | 1.8 | 1 | 74% |
    | LeakyReluFwd | `gan-feat-deep-float16` | 0.0108 | 1.6 | 0 | 65% |
    | LeakyReluFwd | `gan-feat-deep-bfloat16` | 0.0107 | 1.6 | 0 | 65% |
    | LerpFwd | `hidden-state-prefill-float16` | 0.0153 | 1.6 | 0 | 68% |
    | LerpFwd | `hidden-state-prefill-bfloat16` | 0.0153 | 1.6 | 0 | 68% |
    | LerpFwd | `hidden-state-prefill-float32` | 0.0273 | 0.9 | 0 | 77% |
    | LerpFwd | `cnn-feat-broadcast-float16` | 0.0173 | 2.2 | 1 | 62% |
    | LerpFwd | `cnn-feat-broadcast-bfloat16` | 0.0172 | 2.2 | 1 | 62% |
    | LerpFwd | `cnn-feat-broadcast-float32` | 0.0271 | 1.4 | 0 | 79% |
    | LerpTensorFwd | `lerp-tensor-fp16-1024x4096` | 0.0116 | 1.1 | 0 | 60% |
    | LerpTensorFwd | `lerp-tensor-bf16-1024x4096` | 0.0115 | 1.1 | 0 | 61% |
    | LerpTensorFwd | `lerp-tensor-fp32-1024x4096` | 0.0202 | 0.6 | 0 | 69% |
    | LerpTensorFwd | `lerp-tensor-fp16-1024x10240` | 0.0242 | 1.3 | 0 | 72% |
    | LerpTensorFwd | `lerp-tensor-fp16-1024x11008` | 0.0256 | 1.3 | 0 | 74% |
    | LerpTensorFwd | `elementwise-16M-float16` | 0.0358 | 1.4 | 0 | 78% |
    | LerpTensorFwd | `elementwise-16M-bfloat16` | 0.036 | 1.4 | 0 | 78% |
    | LerpTensorFwd | `elementwise-16M-float32` | 0.0852 | 0.6 | 0 | 66% |
    | LerpTensorFwd | `elementwise-256M-float16` | 0.9158 | 0.9 | 0 | 49% |
    | LerpTensorFwd | `elementwise-256M-bfloat16` | 0.9053 | 0.9 | 0 | 49% |
    | Log1pFwd | `elementwise-16M-float16` | 0.029 | 1.2 | 1 | 48% |
    | Log1pFwd | `elementwise-16M-bfloat16` | 0.0299 | 1.1 | 0 | 47% |
    | Log1pFwd | `elementwise-16M-float32` | 0.0387 | 0.9 | 0 | 72% |
    | Log1pFwd | `elementwise-256M-float16` | 0.5489 | 1.0 | 0 | 41% |
    | Log1pFwd | `elementwise-256M-bfloat16` | 0.5742 | 0.9 | 0 | 39% |
    | LogFwd | `elementwise-16M-float16` | 0.0281 | 0.6 | 0 | 50% |
    | LogFwd | `elementwise-16M-bfloat16` | 0.0295 | 0.6 | 0 | 48% |
    | LogFwd | `elementwise-16M-float32` | 0.0368 | 0.5 | 0 | 76% |
    | LogFwd | `elementwise-256M-float16` | 0.5366 | 0.5 | 0 | 42% |
    | LogFwd | `elementwise-256M-bfloat16` | 0.5651 | 0.5 | 0 | 40% |
    | LogicalAndFwd | `hidden-state-prefill-bool` | 0.0512 | 0.5 | 1 | 10% |
    | LogicalAndFwd | `hidden-state-prefill-float16` | 0.0367 | 0.7 | 1 | 24% |
    | LogicalAndFwd | `hidden-state-prefill-bfloat16` | 0.0367 | 0.7 | 1 | 24% |
    | LogicalAndFwd | `hidden-state-prefill-float32` | 0.0463 | 0.5 | 0 | 34% |
    | LogicalAndFwd | `cnn-feat-broadcast-bool` | 0.0769 | 0.5 | 2 | 7% |
    | LogicalAndFwd | `cnn-feat-broadcast-float16` | 0.0505 | 0.8 | 1 | 16% |
    | LogicalAndFwd | `cnn-feat-broadcast-bfloat16` | 0.0503 | 0.8 | 1 | 16% |
    | LogicalAndFwd | `cnn-feat-broadcast-float32` | 0.0575 | 0.7 | 1 | 23% |
    | LogicalNotFwd | `elementwise-16M-bool` | 0.0581 | 0.3 | 0 | 12% |
    | LogicalNotFwd | `elementwise-16M-float16` | 0.067 | 0.2 | 0 | 16% |
    | LogicalNotFwd | `elementwise-16M-float32` | 0.0732 | 0.2 | 0 | 24% |
    | LogicalNotFwd | `elementwise-256M-bool` | 1.0317 | 0.3 | 0 | 11% |
    | LogicalOrFwd | `hidden-state-prefill-bool` | 0.0512 | 0.5 | 1 | 10% |
    | LogicalOrFwd | `hidden-state-prefill-float16` | 0.0367 | 0.7 | 1 | 24% |
    | LogicalOrFwd | `hidden-state-prefill-bfloat16` | 0.0367 | 0.7 | 1 | 24% |
    | LogicalOrFwd | `hidden-state-prefill-float32` | 0.0462 | 0.5 | 0 | 34% |
    | LogicalOrFwd | `cnn-feat-broadcast-bool` | 0.075 | 0.5 | 2 | 7% |
    | LogicalOrFwd | `cnn-feat-broadcast-float16` | 0.0504 | 0.8 | 1 | 16% |
    | LogicalOrFwd | `cnn-feat-broadcast-bfloat16` | 0.0504 | 0.8 | 1 | 16% |
    | LogicalOrFwd | `cnn-feat-broadcast-float32` | 0.0575 | 0.7 | 1 | 23% |
    | LtFwd | `hidden-state-prefill-float16` | 0.0367 | 0.2 | 0 | 24% |
    | LtFwd | `hidden-state-prefill-bfloat16` | 0.0367 | 0.2 | 0 | 24% |
    | LtFwd | `hidden-state-prefill-float32` | 0.0462 | 0.2 | 0 | 34% |
    | LtFwd | `cnn-feat-broadcast-float16` | 0.0507 | 0.2 | 0 | 16% |
    | LtFwd | `cnn-feat-broadcast-bfloat16` | 0.0504 | 0.2 | 0 | 16% |
    | LtFwd | `cnn-feat-broadcast-float32` | 0.0575 | 0.2 | 0 | 23% |
    | MaskedFillFwd | `elementwise-16M-float16` | 0.0232 | 0.7 | 0 | 75% |
    | MaskedFillFwd | `elementwise-16M-bfloat16` | 0.0232 | 0.7 | 0 | 75% |
    | MaskedFillFwd | `elementwise-16M-float32` | 0.0388 | 0.4 | 0 | 81% |
    | MaskedFillFwd | `elementwise-256M-float16` | 0.5526 | 0.5 | 0 | 51% |
    | MaskedFillFwd | `elementwise-256M-bfloat16` | 0.5526 | 0.5 | 0 | 51% |
    | MaskedFillFwd | `elementwise-16M-float16` | 0.0232 | 0.7 | 0 | 75% |
    | MaskedFillFwd | `elementwise-16M-bfloat16` | 0.0232 | 0.7 | 0 | 75% |
    | MaskedFillFwd | `elementwise-16M-float32` | 0.0389 | 0.4 | 0 | 81% |
    | MaskedFillFwd | `elementwise-256M-float16` | 0.4419 | 0.6 | 0 | 63% |
    | MaskedFillFwd | `elementwise-256M-bfloat16` | 0.4419 | 0.6 | 0 | 63% |
    | MaskedFillScalarFwd | `elementwise-16M-float16` | 0.0232 | 0.7 | 0 | 75% |
    | MaskedFillScalarFwd | `elementwise-16M-bfloat16` | 0.0232 | 0.7 | 0 | 75% |
    | MaskedFillScalarFwd | `elementwise-16M-float32` | 0.0389 | 0.4 | 0 | 81% |
    | MaskedFillScalarFwd | `elementwise-256M-float16` | 0.5602 | 0.5 | 0 | 50% |
    | MaskedFillScalarFwd | `elementwise-256M-bfloat16` | 0.5596 | 0.5 | 0 | 50% |
    | MaskedFillScalarFwd | `shape0-dtype0` | 0.0078 | 0.5 | 0 | 56% |
    | MaskedFillScalarFwd | `shape1-dtype1` | 0.0077 | 0.5 | 0 | 56% |
    | MaskedFillScalarFwd | `shape2-dtype2` | 0.0121 | 0.3 | 0 | 65% |
    | MaskedFillScalarFwd | `shape3-dtype3` | 0.0159 | 0.7 | 0 | 69% |
    | MaskedFillScalarFwd | `shape4-dtype4` | 0.0159 | 0.7 | 0 | 69% |
    | MaskedFillScalarFwd | `shape5-dtype5` | 0.0256 | 0.4 | 0 | 77% |
    | MaskedFillScalarFwd | `shape6-dtype6` | 0.0166 | 0.7 | 0 | 71% |
    | MaskedFillScalarFwd | `shape7-dtype7` | 0.0166 | 0.7 | 0 | 71% |
    | MaskedFillScalarFwd | `shape8-dtype8` | 0.0274 | 0.4 | 0 | 77% |
    | MaximumFwd | `hidden-state-prefill-float16` | 0.0156 | 0.5 | 0 | 67% |
    | MaximumFwd | `hidden-state-prefill-bfloat16` | 0.0156 | 0.5 | 0 | 67% |
    | MaximumFwd | `hidden-state-prefill-float32` | 0.0274 | 0.3 | 0 | 77% |
    | MaximumFwd | `cnn-feat-broadcast-float16` | 0.0401 | 0.3 | 0 | 27% |
    | MaximumFwd | `cnn-feat-broadcast-bfloat16` | 0.0403 | 0.3 | 0 | 27% |
    | MaximumFwd | `cnn-feat-broadcast-float32` | 0.0306 | 0.4 | 0 | 70% |
    | MinimumFwd | `hidden-state-prefill-float16` | 0.0156 | 0.5 | 0 | 67% |
    | MinimumFwd | `hidden-state-prefill-bfloat16` | 0.0156 | 0.5 | 0 | 67% |
    | MinimumFwd | `hidden-state-prefill-float32` | 0.0273 | 0.3 | 0 | 77% |
    | MinimumFwd | `cnn-feat-broadcast-float16` | 0.0402 | 0.3 | 0 | 27% |
    | MinimumFwd | `cnn-feat-broadcast-bfloat16` | 0.0401 | 0.3 | 0 | 27% |
    | MinimumFwd | `cnn-feat-broadcast-float32` | 0.0306 | 0.4 | 0 | 70% |
    | MishFwd | `dtype0-mish` | 0.0141 | 1.2 | 1 | 25% |
    | MishFwd | `yolo-p3-float16` | 0.1024 | 1.0 | 1 | 21% |
    | MishFwd | `yolo-p3-bfloat16` | 0.1019 | 1.0 | 1 | 21% |
    | MishFwd | `yolo-p4-float16` | 0.0421 | 1.2 | 1 | 26% |
    | MishFwd | `yolo-p4-bfloat16` | 0.0408 | 1.3 | 1 | 27% |
    | MulFwd | `hidden-state-prefill-float16` | 0.0153 | 0.6 | 0 | 69% |
    | MulFwd | `hidden-state-prefill-bfloat16` | 0.0153 | 0.6 | 0 | 68% |
    | MulFwd | `hidden-state-prefill-float32` | 0.0273 | 0.3 | 0 | 77% |
    | MulFwd | `cnn-feat-broadcast-float16` | 0.0173 | 0.7 | 0 | 62% |
    | MulFwd | `cnn-feat-broadcast-bfloat16` | 0.0172 | 0.8 | 0 | 62% |
    | MulFwd | `cnn-feat-broadcast-float32` | 0.0271 | 0.5 | 0 | 79% |
    | NanToNumFwd | `elementwise-16M-float16` | 0.0195 | 5.2 | 2 | 72% |
    | NanToNumFwd | `elementwise-16M-bfloat16` | 0.0196 | 5.1 | 1 | 71% |
    | NanToNumFwd | `elementwise-16M-float32` | 0.0374 | 2.7 | 1 | 75% |
    | NanToNumFwd | `elementwise-256M-float16` | 0.4814 | 3.4 | 2 | 46% |
    | NanToNumFwd | `elementwise-256M-bfloat16` | 0.4815 | 3.3 | 1 | 46% |
    | NeFwd | `hidden-state-prefill-float16` | 0.0369 | 0.2 | 0 | 24% |
    | NeFwd | `hidden-state-prefill-bfloat16` | 0.0367 | 0.2 | 0 | 24% |
    | NeFwd | `hidden-state-prefill-float32` | 0.0462 | 0.2 | 0 | 34% |
    | NeFwd | `cnn-feat-broadcast-float16` | 0.0504 | 0.2 | 0 | 16% |
    | NeFwd | `cnn-feat-broadcast-bfloat16` | 0.0507 | 0.2 | 0 | 16% |
    | NeFwd | `cnn-feat-broadcast-float32` | 0.0575 | 0.2 | 0 | 23% |
    | NegFwd | `elementwise-16M-float16` | 0.0187 | 0.9 | 0 | 75% |
    | NegFwd | `elementwise-16M-bfloat16` | 0.0187 | 0.9 | 0 | 75% |
    | NegFwd | `elementwise-16M-float32` | 0.0343 | 0.5 | 0 | 81% |
    | NegFwd | `elementwise-256M-float16` | 0.3336 | 0.8 | 0 | 67% |
    | NegFwd | `elementwise-256M-bfloat16` | 0.3336 | 0.8 | 0 | 67% |
    | PowFwd | `hidden-state-prefill-float16` | 0.0368 | 0.7 | 0 | 29% |
    | PowFwd | `hidden-state-prefill-bfloat16` | 0.0379 | 0.7 | 0 | 28% |
    | PowFwd | `hidden-state-prefill-float32` | 0.039 | 0.7 | 0 | 54% |
    | PowFwd | `cnn-feat-broadcast-float16` | 0.0618 | 0.6 | 1 | 17% |
    | PowFwd | `cnn-feat-broadcast-bfloat16` | 0.0653 | 0.6 | 1 | 16% |
    | PowFwd | `cnn-feat-broadcast-float32` | 0.0631 | 0.6 | 0 | 34% |
    | PreluFwd | `cnn-feat-per-channel-float16` | 0.0151 | 1.7 | 0 | 71% |
    | PreluFwd | `cnn-feat-per-channel-bfloat16` | 0.0152 | 1.7 | 1 | 71% |
    | PreluFwd | `cnn-feat-per-channel-deep-float16` | 0.0088 | 1.5 | 1 | 61% |
    | PreluFwd | `cnn-feat-per-channel-deep-bfloat16` | 0.0088 | 1.5 | 0 | 61% |
    | PreluFwd | `shape0-128-dtype0` | 0.0021 | 0.1 | 0 | 5% |
    | PreluFwd | `shape1-128-dtype1` | 0.0021 | 0.1 | 0 | 5% |
    | PreluFwd | `shape2-128-dtype2` | 0.0022 | 0.1 | 0 | 10% |
    | PreluFwd | `shape3-4096-dtype3` | 0.0064 | 0.7 | 0 | 55% |
    | PreluFwd | `shape4-4096-dtype4` | 0.0064 | 0.7 | 0 | 55% |
    | PreluFwd | `shape5-4096-dtype5` | 0.0106 | 0.4 | 0 | 66% |
    | PreluFwd | `shape6-10240-dtype6` | 0.0128 | 0.8 | 0 | 68% |
    | PreluFwd | `shape7-10240-dtype7` | 0.0128 | 0.8 | 0 | 68% |
    | PreluFwd | `shape8-10240-dtype8` | 0.0226 | 0.5 | 0 | 77% |
    | PreluFwd | `shape9-11008-dtype9` | 0.0135 | 0.8 | 0 | 70% |
    | PreluFwd | `shape10-11008-dtype10` | 0.0135 | 0.8 | 0 | 70% |
    | PreluFwd | `shape11-11008-dtype11` | 0.0241 | 0.5 | 0 | 78% |
    | ReciprocalFwd | `elementwise-16M-float16` | 0.0201 | 0.8 | 0 | 69% |
    | ReciprocalFwd | `elementwise-16M-bfloat16` | 0.02 | 0.8 | 0 | 70% |
    | ReciprocalFwd | `elementwise-16M-float32` | 0.0351 | 0.5 | 0 | 80% |
    | ReciprocalFwd | `elementwise-256M-float16` | 0.3604 | 0.7 | 0 | 62% |
    | ReciprocalFwd | `elementwise-256M-bfloat16` | 0.3612 | 0.7 | 0 | 62% |
    | ReluFwd | `shape0-dtype0` | 0.0028 | – | – | – |
    | ReluFwd | `throughput-fp16` | 0.0065 | 0.7 | 0 | 54% |
    | ReluFwd | `throughput-bf16` | 0.0065 | 0.7 | 0 | 54% |
    | ReluFwd | `baseline-fp32` | 0.0107 | 0.4 | 0 | 65% |
    | ReluFwd | `hidden-state-prefill-float16` | 0.0106 | 0.8 | 0 | 66% |
    | ReluFwd | `hidden-state-prefill-bfloat16` | 0.0106 | 0.8 | 0 | 66% |
    | ReluFwd | `hidden-state-decode-bfloat16` | 0.0018 | – | – | – |
    | RemainderFwd | `hidden-state-prefill-float16` | 0.0156 | 2.1 | 1 | 67% |
    | RemainderFwd | `hidden-state-prefill-bfloat16` | 0.0156 | 2.1 | 1 | 67% |
    | RemainderFwd | `hidden-state-prefill-float32` | 0.0273 | 1.2 | 0 | 77% |
    | RemainderFwd | `cnn-feat-broadcast-float16` | 0.0206 | 2.5 | 1 | 52% |
    | RemainderFwd | `cnn-feat-broadcast-bfloat16` | 0.0205 | 2.5 | 1 | 52% |
    | RemainderFwd | `cnn-feat-broadcast-float32` | 0.028 | 1.8 | 0 | 76% |
    | RoundFwd | `elementwise-16M-float16` | 0.0188 | 0.9 | 0 | 74% |
    | RoundFwd | `elementwise-16M-bfloat16` | 0.0186 | 0.9 | 0 | 75% |
    | RoundFwd | `elementwise-16M-float32` | 0.0344 | 0.5 | 0 | 81% |
    | RoundFwd | `elementwise-256M-float16` | 0.3354 | 0.8 | 0 | 67% |
    | RoundFwd | `elementwise-256M-bfloat16` | 0.3355 | 0.8 | 0 | 67% |
    | RsqrtFwd | `elementwise-16M-float16` | 0.0187 | 0.9 | 0 | 75% |
    | RsqrtFwd | `elementwise-16M-bfloat16` | 0.0187 | 0.9 | 0 | 75% |
    | RsqrtFwd | `elementwise-16M-float32` | 0.0343 | 0.5 | 0 | 81% |
    | RsqrtFwd | `elementwise-256M-float16` | 0.3386 | 0.8 | 0 | 66% |
    | RsqrtFwd | `elementwise-256M-bfloat16` | 0.3388 | 0.8 | 0 | 66% |
    | SeluFwd | `dtype0-selu` | 0.0074 | 2.8 | 1 | 47% |
    | SeluFwd | `snn-fc-float16` | 0.0124 | 3.4 | 1 | 56% |
    | SeluFwd | `snn-fc-bfloat16` | 0.0126 | 3.3 | 1 | 56% |
    | SeluFwd | `snn-fc-wide-float16` | 0.0219 | 3.8 | 1 | 64% |
    | SeluFwd | `snn-fc-wide-bfloat16` | 0.0223 | 3.8 | 1 | 62% |
    | SigmoidFwd | `elementwise-16M-float16` | 0.0293 | 2.3 | 1 | 48% |
    | SigmoidFwd | `elementwise-16M-bfloat16` | 0.0259 | 2.6 | 1 | 54% |
    | SigmoidFwd | `elementwise-16M-float32` | 0.0394 | 1.7 | 0 | 71% |
    | SigmoidFwd | `elementwise-256M-float16` | 0.7985 | 1.3 | 1 | 28% |
    | SigmoidFwd | `elementwise-256M-bfloat16` | 0.6674 | 1.6 | 1 | 34% |
    | SignFwd | `elementwise-16M-float16` | 0.0199 | 1.7 | 1 | 70% |
    | SignFwd | `elementwise-16M-bfloat16` | 0.0198 | 1.7 | 0 | 71% |
    | SignFwd | `elementwise-16M-float32` | 0.0343 | 1.0 | 0 | 81% |
    | SignFwd | `elementwise-256M-float16` | 0.3578 | 1.5 | 0 | 62% |
    | SignFwd | `elementwise-256M-bfloat16` | 0.3794 | 1.4 | 1 | 59% |
    | SiluFwd | `dtype0-silu` | 0.0095 | 2.2 | 1 | 37% |
    | SiluFwd | `llama-3.1-8b-ffn-prefill-float16` | 0.0654 | 2.2 | 1 | 38% |
    | SiluFwd | `llama-3.1-8b-ffn-prefill-bfloat16` | 0.0526 | 2.8 | 1 | 46% |
    | SiluFwd | `llama-3.1-8b-ffn-decode-bfloat16` | 0.0021 | 0.0 | 1 | 1% |
    | SinFwd | `elementwise-16M-float16` | 0.0261 | 0.6 | 0 | 54% |
    | SinFwd | `elementwise-16M-bfloat16` | 0.0265 | 0.6 | 0 | 53% |
    | SinFwd | `elementwise-16M-float32` | 0.0362 | 0.5 | 0 | 77% |
    | SinFwd | `elementwise-256M-float16` | 0.4861 | 0.6 | 0 | 46% |
    | SinFwd | `elementwise-256M-bfloat16` | 0.5109 | 0.5 | 0 | 44% |
    | SoftplusFwd | `dtype0-softplus` | 0.0114 | 1.8 | 1 | 31% |
    | SoftplusFwd | `mlp-hidden-float16` | 0.0197 | 2.1 | 1 | 36% |
    | SoftplusFwd | `mlp-hidden-bfloat16` | 0.0201 | 2.1 | 1 | 35% |
    | SoftplusFwd | `mlp-hidden-wide-float16` | 0.0401 | 2.1 | 1 | 35% |
    | SoftplusFwd | `mlp-hidden-wide-bfloat16` | 0.0408 | 2.1 | 1 | 34% |
    | SqrtFwd | `elementwise-16M-float16` | 0.0196 | 0.9 | 0 | 71% |
    | SqrtFwd | `elementwise-16M-bfloat16` | 0.0196 | 0.8 | 0 | 71% |
    | SqrtFwd | `elementwise-16M-float32` | 0.0348 | 0.5 | 0 | 80% |
    | SqrtFwd | `elementwise-256M-float16` | 0.3541 | 0.8 | 0 | 63% |
    | SqrtFwd | `elementwise-256M-bfloat16` | 0.356 | 0.8 | 0 | 63% |
    | SubFwd | `hidden-state-prefill-float16` | 0.0153 | 1.1 | 0 | 68% |
    | SubFwd | `hidden-state-prefill-bfloat16` | 0.0153 | 1.1 | 0 | 69% |
    | SubFwd | `hidden-state-prefill-float32` | 0.0272 | 0.6 | 0 | 77% |
    | SubFwd | `cnn-feat-broadcast-float16` | 0.0172 | 1.5 | 0 | 62% |
    | SubFwd | `cnn-feat-broadcast-bfloat16` | 0.0172 | 1.5 | 0 | 62% |
    | SubFwd | `cnn-feat-broadcast-float32` | 0.0271 | 0.9 | 0 | 79% |
    | TanhFwd | `elementwise-16M-float16` | 0.0216 | 0.8 | 0 | 65% |
    | TanhFwd | `elementwise-16M-bfloat16` | 0.0219 | 0.8 | 0 | 64% |
    | TanhFwd | `elementwise-16M-float32` | 0.0347 | 0.5 | 0 | 81% |
    | TanhFwd | `elementwise-256M-float16` | 0.5255 | 0.5 | 0 | 43% |
    | TanhFwd | `elementwise-256M-bfloat16` | 0.5418 | 0.5 | 0 | 41% |
    | TruncFwd | `elementwise-16M-float16` | 0.0188 | 0.9 | 0 | 74% |
    | TruncFwd | `elementwise-16M-bfloat16` | 0.0187 | 0.9 | 0 | 75% |
    | TruncFwd | `elementwise-16M-float32` | 0.0343 | 0.5 | 0 | 81% |
    | TruncFwd | `elementwise-256M-float16` | 0.3505 | 0.8 | 0 | 64% |
    | TruncFwd | `elementwise-256M-bfloat16` | 0.3559 | 0.8 | 0 | 63% |
    | WhereFwd | `elementwise-16M-float16` | 0.0318 | 0.5 | 0 | 77% |
    | WhereFwd | `elementwise-16M-bfloat16` | 0.0317 | 0.5 | 0 | 77% |
    | WhereFwd | `elementwise-16M-float32` | 0.0626 | 0.3 | 0 | 73% |
    | WhereFwd | `elementwise-256M-float16` | 0.7897 | 0.3 | 0 | 50% |
    | WhereFwd | `elementwise-256M-bfloat16` | 0.7897 | 0.3 | 0 | 50% |
    | WhereFwd | `shape0-dtype0` | 0.0103 | 0.4 | 0 | 60% |
    | WhereFwd | `shape1-dtype1` | 0.0103 | 0.4 | 0 | 59% |
    | WhereFwd | `shape2-dtype2` | 0.0169 | 0.2 | 0 | 67% |
    | WhereFwd | `shape3-dtype3` | 0.0215 | 0.5 | 0 | 71% |
    | WhereFwd | `shape4-dtype4` | 0.0215 | 0.5 | 0 | 71% |
    | WhereFwd | `shape5-dtype5` | 0.0357 | 0.3 | 0 | 80% |
    | WhereFwd | `shape6-dtype6` | 0.0227 | 0.5 | 0 | 73% |
    | WhereFwd | `shape7-dtype7` | 0.0227 | 0.5 | 0 | 73% |
    | WhereFwd | `shape8-dtype8` | 0.0384 | 0.3 | 0 | 79% |
    | alibi | `alibi-512-64-dtype0` | 0.0106 | 1.6 | 0 | 66% |
    | alibi | `alibi-512-64-dtype1` | 0.0106 | 1.6 | 0 | 66% |
    | alibi | `alibi-512-64-dtype2` | 0.016 | 1.1 | 0 | 87% |
    | alibi | `alibi-2048-64-dtype3` | 0.202 | 1.3 | 0 | 55% |
    | alibi | `alibi-2048-64-dtype4` | 0.2017 | 1.3 | 0 | 55% |
    | alibi | `alibi-2048-64-dtype5` | 0.3271 | 0.8 | 0 | 68% |
    | alibi | `alibi-4096-128-dtype6` | 1.6576 | 1.3 | 1 | 54% |
    | alibi | `alibi-4096-128-dtype7` | 1.6563 | 1.3 | 1 | 54% |
    | alibi | `alibi-4096-128-dtype8` | 2.5987 | 0.8 | 0 | 69% |
    | bitwise_and | `bitwise_and-shape0-BitwiseAndFwdOp-bitwise_and` | 0.0154 | 0.3 | 0 | 68% |
    | bitwise_and | `bitwise_and-shape1-BitwiseAndFwdOp-bitwise_and` | 0.0336 | 0.3 | 0 | 78% |
    | bitwise_or | `bitwise_or-shape2-BitwiseOrFwdOp-bitwise_or` | 0.0157 | 0.3 | 0 | 67% |
    | bitwise_xor | `bitwise_xor-shape3-BitwiseXorFwdOp-bitwise_xor` | 0.0156 | 0.3 | 0 | 67% |
    | clamp | `clamp-shape36-dtype36` | 0.0064 | 0.7 | 0 | 55% |
    | clamp | `clamp-shape37-dtype37` | 0.0064 | 0.7 | 0 | 54% |
    | clamp | `clamp-shape38-dtype38` | 0.0107 | 0.4 | 0 | 65% |
    | clamp | `clamp-shape39-dtype39` | 0.0129 | 0.8 | 0 | 68% |
    | clamp | `clamp-shape40-dtype40` | 0.0128 | 0.8 | 0 | 68% |
    | clamp | `clamp-shape41-dtype41` | 0.0225 | 0.5 | 0 | 78% |
    | clamp | `clamp-shape42-dtype42` | 0.0134 | 0.8 | 0 | 70% |
    | clamp | `clamp-shape43-dtype43` | 0.0135 | 0.8 | 0 | 70% |
    | clamp | `clamp-shape44-dtype44` | 0.0239 | 0.5 | 0 | 79% |
    | cmp_eq | `eq-shape0-dtype0-eq` | 0.0209 | 0.2 | 0 | 21% |
    | cmp_eq | `eq-shape1-dtype1-eq` | 0.0447 | 0.2 | 0 | 24% |
    | cmp_ge | `ge-shape5-dtype5-ge` | 0.0209 | 0.2 | 0 | 21% |
    | cmp_gt | `gt-shape3-dtype3-gt` | 0.0208 | 0.2 | 0 | 21% |
    | cmp_le | `le-shape6-dtype6-le` | 0.0209 | 0.2 | 0 | 21% |
    | cmp_lt | `lt-shape4-dtype4-lt` | 0.0209 | 0.2 | 0 | 21% |
    | cmp_ne | `ne-shape2-dtype2-ne` | 0.0209 | 0.2 | 0 | 21% |
    | div | `div-shape6-dtype6-output_dtype6-DivFwdOp-div-_positive_pair` | 0.0091 | 0.5 | 0 | 57% |
    | div | `div-shape7-dtype7-output_dtype7-DivFwdOp-div-_positive_pair` | 0.0185 | 0.6 | 0 | 71% |
    | div | `div-shape8-dtype8-output_dtype8-DivFwdOp-div-_positive_pair` | 0.0193 | 0.6 | 0 | 73% |
    | div_bcast | `div-a_shape6-b_shape6-dtype6-DivFwdOp-div-_positive_broadcast_pair` | 0.0074 | 0.6 | 0 | 47% |
    | div_bcast | `div-a_shape7-b_shape7-dtype7-DivFwdOp-div-_positive_broadcast_pair` | 0.0155 | 0.7 | 0 | 56% |
    | div_bcast | `div-a_shape8-b_shape8-dtype8-DivFwdOp-div-_positive_broadcast_pair` | 0.0167 | 0.7 | 0 | 56% |
    | elu | `elu-shape9-dtype9` | 0.0075 | 0.6 | 0 | 47% |
    | elu | `elu-shape10-dtype10` | 0.0075 | 0.6 | 0 | 47% |
    | elu | `elu-shape11-dtype11` | 0.0108 | 0.4 | 0 | 64% |
    | elu | `elu-shape12-dtype12` | 0.0149 | 0.7 | 0 | 59% |
    | elu | `elu-shape13-dtype13` | 0.015 | 0.7 | 0 | 58% |
    | elu | `elu-shape14-dtype14` | 0.0233 | 0.5 | 0 | 75% |
    | elu | `elu-shape15-dtype15` | 0.0158 | 0.7 | 0 | 60% |
    | elu | `elu-shape16-dtype16` | 0.0158 | 0.7 | 0 | 60% |
    | elu | `elu-shape17-dtype17` | 0.0248 | 0.5 | 0 | 76% |
    | floor_divide | `floor_divide-shape13-dtype13-output_dtype13-FloorDivideFwdOp-floor_divide-_positive_pair` | 0.0091 | 0.5 | 0 | 57% |
    | floor_divide | `floor_divide-shape14-dtype14-output_dtype14-FloorDivideFwdOp-floor_divide-_positive_pair` | 0.0186 | 0.6 | 0 | 70% |
    | gelu_and_mul | `gelu_and_mul-1024-4096-dtype0-GeluAndMulFwdOp` | 0.0109 | 0.8 | 0 | 48% |
    | gelu_and_mul | `gelu_and_mul-1024-10240-dtype1-GeluAndMulFwdOp` | 0.0237 | 0.9 | 0 | 55% |
    | gelu_and_mul | `gelu_and_mul-1024-11008-dtype2-GeluAndMulFwdOp` | 0.026 | 0.9 | 0 | 54% |
    | gelu_and_mul_strategy | `gelu_and_mul-1024-4096-dtype18-GeluAndMulFwdOp-direct` | 0.0189 | 0.4 | 0 | 28% |
    | gelu_and_mul_strategy | `gelu_and_mul-1024-4096-dtype19-GeluAndMulFwdOp-explicit_parallel` | 0.0109 | 0.8 | 0 | 48% |
    | gelu_and_mul_strategy | `gelu_and_mul-1024-4096-dtype20-GeluAndMulFwdOp-direct` | 0.0189 | 0.4 | 0 | 28% |
    | gelu_and_mul_strategy | `gelu_and_mul-1024-4096-dtype21-GeluAndMulFwdOp-explicit_parallel` | 0.0112 | 0.8 | 0 | 47% |
    | gelu_and_mul_strategy | `gelu_and_mul-1024-4096-dtype22-GeluAndMulFwdOp-direct` | 0.0207 | 0.4 | 0 | 51% |
    | gelu_and_mul_strategy | `gelu_and_mul-1024-4096-dtype23-GeluAndMulFwdOp-explicit_parallel` | 0.0158 | 0.5 | 0 | 66% |
    | gelu_and_mul_strategy | `gelu_and_mul-1024-11008-dtype24-GeluAndMulFwdOp-direct` | 0.0556 | 0.4 | 0 | 25% |
    | gelu_and_mul_strategy | `gelu_and_mul-1024-11008-dtype25-GeluAndMulFwdOp-explicit_parallel` | 0.0262 | 0.9 | 0 | 54% |
    | gelu_and_mul_strategy | `gelu_and_mul-1024-11008-dtype26-GeluAndMulFwdOp-direct` | 0.0536 | 0.4 | 0 | 26% |
    | gelu_and_mul_strategy | `gelu_and_mul-1024-11008-dtype27-GeluAndMulFwdOp-explicit_parallel` | 0.0264 | 0.9 | 0 | 54% |
    | gelu_and_mul_strategy | `gelu_and_mul-1024-11008-dtype28-GeluAndMulFwdOp-direct` | 0.0613 | 0.4 | 0 | 46% |
    | gelu_and_mul_strategy | `gelu_and_mul-1024-11008-dtype29-GeluAndMulFwdOp-explicit_parallel` | 0.0698 | 0.3 | 0 | 40% |
    | gelu_and_mul_strategy | `gelu_and_mul-4096-4096-dtype30-GeluAndMulFwdOp-direct` | 0.0928 | 0.4 | 0 | 22% |
    | gelu_and_mul_strategy | `gelu_and_mul-4096-4096-dtype31-GeluAndMulFwdOp-explicit_parallel` | 0.0381 | 0.9 | 0 | 55% |
    | gelu_and_mul_strategy | `gelu_and_mul-4096-4096-dtype32-GeluAndMulFwdOp-direct` | 0.0946 | 0.3 | 0 | 22% |
    | gelu_and_mul_strategy | `gelu_and_mul-4096-4096-dtype33-GeluAndMulFwdOp-explicit_parallel` | 0.0397 | 0.8 | 0 | 53% |
    | gelu_and_mul_strategy | `gelu_and_mul-4096-4096-dtype34-GeluAndMulFwdOp-direct` | 0.1031 | 0.3 | 0 | 41% |
    | gelu_and_mul_strategy | `gelu_and_mul-4096-4096-dtype35-GeluAndMulFwdOp-explicit_parallel` | 0.0654 | 0.5 | 0 | 64% |
    | gelu_tanh_and_mul | `gelu_tanh_and_mul-1024-4096-dtype3-GeluTanhAndMulFwdOp` | 0.0096 | 0.9 | 0 | 55% |
    | gelu_tanh_and_mul | `gelu_tanh_and_mul-1024-10240-dtype4-GeluTanhAndMulFwdOp` | 0.0209 | 1.0 | 0 | 63% |
    | gelu_tanh_and_mul | `gelu_tanh_and_mul-1024-11008-dtype5-GeluTanhAndMulFwdOp` | 0.0225 | 1.0 | 0 | 63% |
    | gelu_tanh_and_mul_strategy | `gelu_tanh_and_mul-1024-4096-dtype36-GeluTanhAndMulFwdOp-direct` | 0.0184 | 0.5 | 0 | 29% |
    | gelu_tanh_and_mul_strategy | `gelu_tanh_and_mul-1024-4096-dtype37-GeluTanhAndMulFwdOp-explicit_parallel` | 0.0096 | 0.9 | 0 | 55% |
    | gelu_tanh_and_mul_strategy | `gelu_tanh_and_mul-1024-4096-dtype38-GeluTanhAndMulFwdOp-direct` | 0.0182 | 0.5 | 0 | 29% |
    | gelu_tanh_and_mul_strategy | `gelu_tanh_and_mul-1024-4096-dtype39-GeluTanhAndMulFwdOp-explicit_parallel` | 0.0097 | 0.9 | 0 | 54% |
    | gelu_tanh_and_mul_strategy | `gelu_tanh_and_mul-1024-4096-dtype40-GeluTanhAndMulFwdOp-direct` | 0.0204 | 0.4 | 0 | 51% |
    | gelu_tanh_and_mul_strategy | `gelu_tanh_and_mul-1024-4096-dtype41-GeluTanhAndMulFwdOp-explicit_parallel` | 0.0156 | 0.5 | 0 | 67% |
    | gelu_tanh_and_mul_strategy | `gelu_tanh_and_mul-1024-11008-dtype42-GeluTanhAndMulFwdOp-direct` | 0.0531 | 0.4 | 0 | 26% |
    | gelu_tanh_and_mul_strategy | `gelu_tanh_and_mul-1024-11008-dtype43-GeluTanhAndMulFwdOp-explicit_parallel` | 0.0224 | 1.0 | 0 | 63% |
    | gelu_tanh_and_mul_strategy | `gelu_tanh_and_mul-1024-11008-dtype44-GeluTanhAndMulFwdOp-direct` | 0.0539 | 0.4 | 0 | 26% |
    | gelu_tanh_and_mul_strategy | `gelu_tanh_and_mul-1024-11008-dtype45-GeluTanhAndMulFwdOp-explicit_parallel` | 0.0224 | 1.0 | 0 | 63% |
    | gelu_tanh_and_mul_strategy | `gelu_tanh_and_mul-1024-11008-dtype46-GeluTanhAndMulFwdOp-direct` | 0.06 | 0.4 | 0 | 47% |
    | gelu_tanh_and_mul_strategy | `gelu_tanh_and_mul-1024-11008-dtype47-GeluTanhAndMulFwdOp-explicit_parallel` | 0.0363 | 0.6 | 0 | 78% |
    | gelu_tanh_and_mul_strategy | `gelu_tanh_and_mul-4096-4096-dtype48-GeluTanhAndMulFwdOp-direct` | 0.0904 | 0.4 | 0 | 23% |
    | gelu_tanh_and_mul_strategy | `gelu_tanh_and_mul-4096-4096-dtype49-GeluTanhAndMulFwdOp-explicit_parallel` | 0.0315 | 1.1 | 0 | 66% |
    | gelu_tanh_and_mul_strategy | `gelu_tanh_and_mul-4096-4096-dtype50-GeluTanhAndMulFwdOp-direct` | 0.0902 | 0.4 | 0 | 23% |
    | gelu_tanh_and_mul_strategy | `gelu_tanh_and_mul-4096-4096-dtype51-GeluTanhAndMulFwdOp-explicit_parallel` | 0.0319 | 1.1 | 0 | 66% |
    | gelu_tanh_and_mul_strategy | `gelu_tanh_and_mul-4096-4096-dtype52-GeluTanhAndMulFwdOp-direct` | 0.1009 | 0.3 | 0 | 41% |
    | gelu_tanh_and_mul_strategy | `gelu_tanh_and_mul-4096-4096-dtype53-GeluTanhAndMulFwdOp-explicit_parallel` | 0.0615 | 0.6 | 0 | 68% |
    | hardtanh | `hardtanh-shape18-dtype18` | 0.0064 | 0.7 | 0 | 54% |
    | hardtanh | `hardtanh-shape19-dtype19` | 0.0064 | 0.7 | 0 | 55% |
    | hardtanh | `hardtanh-shape20-dtype20` | 0.0106 | 0.4 | 0 | 66% |
    | hardtanh | `hardtanh-shape21-dtype21` | 0.0129 | 0.8 | 0 | 68% |
    | hardtanh | `hardtanh-shape22-dtype22` | 0.0129 | 0.8 | 0 | 68% |
    | hardtanh | `hardtanh-shape23-dtype23` | 0.0225 | 0.5 | 0 | 78% |
    | hardtanh | `hardtanh-shape24-dtype24` | 0.0135 | 0.8 | 0 | 70% |
    | hardtanh | `hardtanh-shape25-dtype25` | 0.0135 | 0.8 | 0 | 70% |
    | hardtanh | `hardtanh-shape26-dtype26` | 0.024 | 0.5 | 0 | 78% |
    | leaky_relu | `leaky_relu-shape0-dtype0` | 0.0065 | 0.7 | 0 | 54% |
    | leaky_relu | `leaky_relu-shape1-dtype1` | 0.0065 | 0.6 | 0 | 54% |
    | leaky_relu | `leaky_relu-shape2-dtype2` | 0.0107 | 0.4 | 0 | 66% |
    | leaky_relu | `leaky_relu-shape3-dtype3` | 0.0127 | 0.8 | 0 | 69% |
    | leaky_relu | `leaky_relu-shape4-dtype4` | 0.0127 | 0.8 | 0 | 69% |
    | leaky_relu | `leaky_relu-shape5-dtype5` | 0.0225 | 0.5 | 0 | 77% |
    | leaky_relu | `leaky_relu-shape6-dtype6` | 0.0135 | 0.8 | 0 | 70% |
    | leaky_relu | `leaky_relu-shape7-dtype7` | 0.0135 | 0.8 | 0 | 70% |
    | leaky_relu | `leaky_relu-shape8-dtype8` | 0.0238 | 0.5 | 0 | 79% |
    | lerp | `lerp-shape15-dtype15-output_dtype15-LerpFwdOp-<lambda>-_randn_pair` | 0.0088 | 0.5 | 0 | 60% |
    | lerp | `lerp-shape16-dtype16-output_dtype16-LerpFwdOp-<lambda>-_randn_pair` | 0.0184 | 0.6 | 0 | 71% |
    | logical_and | `logical_and-shape0-dtype0-LogicalAndFwdOp-logical_and` | 0.0209 | 0.2 | 0 | 21% |
    | logical_and | `logical_and-shape1-dtype1-LogicalAndFwdOp-logical_and` | 0.1185 | 0.1 | 0 | 9% |
    | logical_or | `logical_or-shape2-dtype2-LogicalOrFwdOp-logical_or` | 0.0209 | 0.2 | 0 | 21% |
    | logical_or | `logical_or-shape3-dtype3-LogicalOrFwdOp-logical_or` | 0.0447 | 0.2 | 0 | 24% |
    | mul | `mul-shape3-dtype3-output_dtype3-MulFwdOp-mul-_randn_pair` | 0.0089 | 0.5 | 0 | 59% |
    | mul | `mul-shape4-dtype4-output_dtype4-MulFwdOp-mul-_randn_pair` | 0.0184 | 0.6 | 0 | 71% |
    | mul | `mul-shape5-dtype5-output_dtype5-MulFwdOp-mul-_randn_pair` | 0.0193 | 0.6 | 0 | 73% |
    | mul_bcast | `mul-a_shape3-b_shape3-dtype3-MulFwdOp-mul-_randn_broadcast_pair` | 0.0069 | 0.6 | 0 | 51% |
    | mul_bcast | `mul-a_shape4-b_shape4-dtype4-MulFwdOp-mul-_randn_broadcast_pair` | 0.0142 | 0.7 | 0 | 62% |
    | mul_bcast | `mul-a_shape5-b_shape5-dtype5-MulFwdOp-mul-_randn_broadcast_pair` | 0.0151 | 0.7 | 0 | 62% |
    | nan_to_num | `nan_to_num-shape45-dtype45` | 0.0066 | 0.6 | 0 | 53% |
    | nan_to_num | `nan_to_num-shape46-dtype46` | 0.0066 | 0.6 | 0 | 53% |
    | nan_to_num | `nan_to_num-shape47-dtype47` | 0.0106 | 0.4 | 0 | 66% |
    | nan_to_num | `nan_to_num-shape48-dtype48` | 0.0132 | 0.8 | 0 | 66% |
    | nan_to_num | `nan_to_num-shape49-dtype49` | 0.0131 | 0.8 | 0 | 66% |
    | nan_to_num | `nan_to_num-shape50-dtype50` | 0.0227 | 0.5 | 0 | 77% |
    | nan_to_num | `nan_to_num-shape51-dtype51` | 0.014 | 0.8 | 0 | 67% |
    | nan_to_num | `nan_to_num-shape52-dtype52` | 0.0139 | 0.8 | 0 | 68% |
    | nan_to_num | `nan_to_num-shape53-dtype53` | 0.024 | 0.5 | 0 | 78% |
    | pow | `pow-shape11-dtype11-output_dtype11-PowFwdOp-pow-_positive_pair` | 0.0203 | 0.2 | 0 | 26% |
    | pow | `pow-shape12-dtype12-output_dtype12-PowFwdOp-pow-_positive_pair` | 0.0452 | 0.2 | 0 | 29% |
    | r4_where | `where-4K-fp16` | 0.002 | – | – | – |
    | r4_where | `where-1M-fp16` | 0.0043 | 0.2 | 0 | 35% |
    | r4_where | `where-11M-fp16` | 0.0226 | 0.5 | 0 | 73% |
    | remainder | `remainder-shape9-dtype9-output_dtype9-RemainderFwdOp-remainder-_positive_pair` | 0.0092 | 0.5 | 0 | 57% |
    | remainder | `remainder-shape10-dtype10-output_dtype10-RemainderFwdOp-remainder-_positive_pair` | 0.0186 | 0.6 | 0 | 70% |
    | silu_and_mul_strategy | `silu_and_mul-1024-4096-dtype0-SiluAndMulFwdOp-direct` | 0.0184 | 0.5 | 0 | 29% |
    | silu_and_mul_strategy | `silu_and_mul-1024-4096-dtype1-SiluAndMulFwdOp-explicit_parallel` | 0.0095 | 0.9 | 0 | 55% |
    | silu_and_mul_strategy | `silu_and_mul-1024-4096-dtype2-SiluAndMulFwdOp-direct` | 0.0184 | 0.5 | 0 | 29% |
    | silu_and_mul_strategy | `silu_and_mul-1024-4096-dtype3-SiluAndMulFwdOp-explicit_parallel` | 0.0096 | 0.9 | 0 | 55% |
    | silu_and_mul_strategy | `silu_and_mul-1024-4096-dtype4-SiluAndMulFwdOp-direct` | 0.0209 | 0.4 | 0 | 50% |
    | silu_and_mul_strategy | `silu_and_mul-1024-4096-dtype5-SiluAndMulFwdOp-explicit_parallel` | 0.0155 | 0.5 | 0 | 68% |
    | silu_and_mul_strategy | `silu_and_mul-1024-11008-dtype6-SiluAndMulFwdOp-direct` | 0.0534 | 0.4 | 0 | 26% |
    | silu_and_mul_strategy | `silu_and_mul-1024-11008-dtype7-SiluAndMulFwdOp-explicit_parallel` | 0.0229 | 1.0 | 0 | 62% |
    | silu_and_mul_strategy | `silu_and_mul-1024-11008-dtype8-SiluAndMulFwdOp-direct` | 0.0542 | 0.4 | 0 | 26% |
    | silu_and_mul_strategy | `silu_and_mul-1024-11008-dtype9-SiluAndMulFwdOp-explicit_parallel` | 0.0236 | 0.9 | 0 | 60% |
    | silu_and_mul_strategy | `silu_and_mul-1024-11008-dtype10-SiluAndMulFwdOp-direct` | 0.0624 | 0.4 | 0 | 45% |
    | silu_and_mul_strategy | `silu_and_mul-1024-11008-dtype11-SiluAndMulFwdOp-explicit_parallel` | 0.0364 | 0.6 | 0 | 77% |
    | silu_and_mul_strategy | `silu_and_mul-4096-4096-dtype12-SiluAndMulFwdOp-direct` | 0.0908 | 0.4 | 0 | 23% |
    | silu_and_mul_strategy | `silu_and_mul-4096-4096-dtype13-SiluAndMulFwdOp-explicit_parallel` | 0.031 | 1.1 | 0 | 68% |
    | silu_and_mul_strategy | `silu_and_mul-4096-4096-dtype14-SiluAndMulFwdOp-direct` | 0.0893 | 0.4 | 0 | 24% |
    | silu_and_mul_strategy | `silu_and_mul-4096-4096-dtype15-SiluAndMulFwdOp-explicit_parallel` | 0.0317 | 1.1 | 0 | 66% |
    | silu_and_mul_strategy | `silu_and_mul-4096-4096-dtype16-SiluAndMulFwdOp-direct` | 0.1017 | 0.3 | 0 | 41% |
    | silu_and_mul_strategy | `silu_and_mul-4096-4096-dtype17-SiluAndMulFwdOp-explicit_parallel` | 0.0607 | 0.6 | 0 | 69% |
    | sinusoidal | `sinusoidal-512-256-dtype9` | 0.0037 | 0.0 | 1 | 1% |
    | sinusoidal | `sinusoidal-512-256-dtype10` | 0.0037 | 0.0 | 1 | 1% |
    | sinusoidal | `sinusoidal-512-256-dtype11` | 0.0025 | 0.1 | 0 | 4% |
    | sinusoidal | `sinusoidal-2048-300-dtype12` | 0.005 | 0.1 | 0 | 5% |
    | sinusoidal | `sinusoidal-2048-300-dtype13` | 0.005 | 0.1 | 0 | 5% |
    | sinusoidal | `sinusoidal-2048-300-dtype14` | 0.0039 | 0.2 | 0 | 13% |
    | sinusoidal | `sinusoidal-4096-512-dtype15` | 0.0076 | 0.3 | 1 | 11% |
    | sinusoidal | `sinusoidal-4096-512-dtype16` | 0.0076 | 0.3 | 1 | 11% |
    | sinusoidal | `sinusoidal-4096-512-dtype17` | 0.0078 | 0.3 | 0 | 22% |
    | softplus | `softplus-shape27-dtype27` | 0.0115 | 0.4 | 0 | 30% |
    | softplus | `softplus-shape28-dtype28` | 0.0116 | 0.4 | 0 | 30% |
    | softplus | `softplus-shape29-dtype29` | 0.0125 | 0.3 | 0 | 56% |
    | softplus | `softplus-shape30-dtype30` | 0.0238 | 0.4 | 0 | 37% |
    | softplus | `softplus-shape31-dtype31` | 0.0244 | 0.4 | 0 | 36% |
    | softplus | `softplus-shape32-dtype32` | 0.0277 | 0.4 | 0 | 63% |
    | softplus | `softplus-shape33-dtype33` | 0.0253 | 0.5 | 0 | 37% |
    | softplus | `softplus-shape34-dtype34` | 0.0261 | 0.4 | 0 | 36% |
    | softplus | `softplus-shape35-dtype35` | 0.0294 | 0.4 | 0 | 64% |
    | sub | `sub-shape0-dtype0-output_dtype0-SubFwdOp-sub-_randn_pair` | 0.0089 | 0.5 | 0 | 59% |
    | sub | `sub-shape1-dtype1-output_dtype1-SubFwdOp-sub-_randn_pair` | 0.0183 | 0.6 | 0 | 72% |
    | sub | `sub-shape2-dtype2-output_dtype2-SubFwdOp-sub-_randn_pair` | 0.0193 | 0.6 | 0 | 73% |
    | sub_bcast | `sub-a_shape0-b_shape0-dtype0-SubFwdOp-sub-_randn_broadcast_pair` | 0.0069 | 0.6 | 0 | 51% |
    | sub_bcast | `sub-a_shape1-b_shape1-dtype1-SubFwdOp-sub-_randn_broadcast_pair` | 0.0142 | 0.7 | 0 | 61% |
    | sub_bcast | `sub-a_shape2-b_shape2-dtype2-SubFwdOp-sub-_randn_broadcast_pair` | 0.0151 | 0.8 | 0 | 62% |

## Convolution  <small>(3 ops)</small>

| Op | Tests | Configs | TFLOPS | % roof | vs baseline | Status |
| --- | :---: | ---: | ---: | ---: | --- | :---: |
| [conv2d](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+conv2d&type=code) | – | 14 | 23.6 | 3% | 🔴 3% roof | 🔴 |
| [conv1d](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+conv1d&type=code) | – | 6 | 106.6 | 13% | — | — |
| [conv3d](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+conv3d&type=code) | – | 4 | 45.4 | 5% | — | — |

??? note "Per-config detail"

    | Op | Config | Latency (ms) | TFLOPS | AI | % roof |
    | --- | --- | ---: | ---: | ---: | ---: |
    | conv1d | `convtasnet-pointwise-k1-s1-fp16` | 0.3124 | 107.4 | 170 | 13% |
    | conv1d | `seanet-k3-s1-fp16` | 0.0342 | 94.1 | 254 | 10% |
    | conv1d | `audio-downsample-k5-s2-fp16` | 0.0248 | 105.7 | 160 | 14% |
    | conv1d | `seanet-stem-k7-s1-fp16` | 0.0712 | 211.1 | 586 | 21% |
    | conv1d | `sequence-downsample-k3-s2-bf16` | 0.0175 | 46.1 | 184 | 5% |
    | conv1d | `seanet-k3-s1-d2-fp16` | 0.0224 | 144.0 | 253 | 15% |
    | conv2d | `resnet-3x3-fp16` | 0.0197 | 23.5 | 261 | 2% |
    | conv2d | `stem-3x3-s2-fp16` | 0.0039 | 2.8 | 23 | 2% |
    | conv2d | `stage-transition-3x3-s2-fp16` | 0.0335 | 13.8 | 276 | 1% |
    | conv2d | `highres-3x3-s1-fp16` | 0.5347 | 55.4 | 1384 | 6% |
    | conv2d | `midres-5x5-s1-fp16` | 0.0543 | 23.7 | 789 | 2% |
    | conv2d | `stage-transition-5x5-s2-fp16` | 0.1012 | 12.7 | 423 | 1% |
    | conv2d | `stride2-bf16` | 0.0307 | 1.9 | 94 | 0% |
    | conv2d | `resnet-1x1-fp16` | 0.0049 | 42.3 | 51 | 17% |
    | conv2d | `bottleneck-expand-1x1-fp16` | 0.0044 | 46.8 | 95 | 10% |
    | conv2d | `bottleneck-reduce-1x1-fp16` | 0.0054 | 37.8 | 97 | 8% |
    | conv2d | `late-stage-1x1-fp16` | 0.0063 | 16.3 | 102 | 3% |
    | conv2d | `classifier-1x1-fp16` | 0.0126 | 8.1 | 43 | 4% |
    | conv2d | `resnet-1x1-bf16` | 0.0048 | 42.7 | 51 | 18% |
    | conv2d | `deeplabv3-aspp-3x3-rate12-fp16` | 0.3349 | 28.9 | 722 | 3% |
    | conv3d | `r3d-stem-k3-s1-fp16` | 0.0373 | 55.8 | 77 | 15% |
    | conv3d | `video-stage-downsample-k3-s2-fp16` | 0.0397 | 35.0 | 318 | 4% |
    | conv3d | `unet-encoder-k3-s1-bf16` | 0.6911 | 21.0 | 524 | 2% |
    | conv3d | `3d-unet-aspp-3x3x3-rate6-fp16` | 0.111 | 65.3 | 1305 | 7% |

## Pooling  <small>(3 ops)</small>

| Op | Tests | Configs | TFLOPS | % roof | vs baseline | Status |
| --- | :---: | ---: | ---: | ---: | --- | :---: |
| [avg_pool1d](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+avg_pool1d&type=code) | – | 3 | 0.1 | 4% | 🔴 4% roof | 🔴 |
| [avg_pool2d](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+avg_pool2d&type=code) | – | 3 | 0.3 | 2% | 🔴 2% roof | 🔴 |
| [avg_pool3d](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+avg_pool3d&type=code) | – | 3 | 0.1 | 1% | 🔴 1% roof | 🔴 |

??? note "Per-config detail"

    | Op | Config | Latency (ms) | TFLOPS | AI | % roof |
    | --- | --- | ---: | ---: | ---: | ---: |
    | avg_pool1d | `audio-downsample-fp16-float16` | 0.0321 | 0.1 | 0 | 4% |
    | avg_pool1d | `long-temporal-fp16-float16` | 0.2198 | 0.1 | 0 | 4% |
    | avg_pool1d | `ceil-bf16-bfloat16` | 0.0125 | 0.1 | 1 | 3% |
    | avg_pool2d | `vision-3x3-s2` | 0.0117 | 0.3 | 1 | 7% |
    | avg_pool2d | `vision-5x5-s2` | 0.0163 | 0.3 | 3 | 2% |
    | avg_pool2d | `ceil-divisor-bf16` | 0.0192 | 0.2 | 2 | 2% |
    | avg_pool3d | `video-2x2x2` | 0.0041 | 0.4 | 0 | 18% |
    | avg_pool3d | `ceil-video` | 0.0277 | 0.1 | 1 | 1% |
    | avg_pool3d | `divisor-bf16` | 0.0074 | 0.1 | 1 | 1% |

## Quantization  <small>(2 ops)</small>

| Op | Tests | Configs | TFLOPS | % roof | vs baseline | Status |
| --- | :---: | ---: | ---: | ---: | --- | :---: |
| [FP8Quant](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/fp8_quant.py) | ✅ | 4 | 1.0 | 8% | 🔴 8% roof | 🔴 |
| [FP8LightingIndexer](https://github.com/tile-ai/TileOPs/blob/main/tileops/ops/fp8_lighting_indexer.py) | ✅ | 2 | – | – | — | — |

??? note "Per-config detail"

    | Op | Config | Latency (ms) | TFLOPS | AI | % roof |
    | --- | --- | ---: | ---: | ---: | ---: |
    | FP8LightingIndexer | `default-config` | 0.2073 | – | – | – |
    | FP8LightingIndexer | `mid-shape` | 0.1355 | – | – | – |
    | FP8Quant | `mainstream-fp16` | 0.0032 | 1.0 | 3 | 7% |
    | FP8Quant | `mainstream-bf16` | 0.0032 | 1.0 | 3 | 7% |
    | FP8Quant | `wider-index` | 0.0043 | 0.7 | 2 | 10% |
    | FP8Quant | `long-sequence` | 0.0033 | 1.0 | 2 | 13% |

## Top-k  <small>(1 ops)</small>

| Op | Tests | Configs | TFLOPS | % roof | vs baseline | Status |
| --- | :---: | ---: | ---: | ---: | --- | :---: |
| [fused_topk](https://github.com/tile-ai/TileOPs/search?q=repo%3Atile-ai%2FTileOPs+fused_topk&type=code) | – | 12 | 0.1 | 1% | 🟡 0.84× vllm | 🟡 |

??? note "Per-config detail"

    | Op | Config | Latency (ms) | TFLOPS | AI | % roof |
    | --- | --- | ---: | ---: | ---: | ---: |
    | fused_topk | `1-384-8-sigmoid-True` | 0.0148 | – | – | – |
    | fused_topk | `32-384-8-sigmoid-True` | 0.0189 | 0.0 | – | – |
    | fused_topk | `512-384-8-sigmoid-True` | 0.0198 | 0.2 | 9 | 0% |
    | fused_topk | `4096-384-8-sigmoid-True` | 0.0282 | 1.0 | 8 | 2% |
    | fused_topk | `1-256-8-sigmoid-True` | 0.0128 | – | – | – |
    | fused_topk | `32-256-8-sigmoid-True` | 0.0164 | 0.0 | – | – |
    | fused_topk | `512-256-8-sigmoid-True` | 0.0171 | 0.1 | 7 | 0% |
    | fused_topk | `4096-256-8-sigmoid-True` | 0.0223 | 0.8 | 8 | 2% |
    | fused_topk | `1-128-8-softmax-False` | 0.0052 | – | – | – |
    | fused_topk | `32-128-8-softmax-False` | 0.0082 | 0.0 | – | – |
    | fused_topk | `512-128-8-softmax-False` | 0.0086 | 0.1 | 7 | 0% |
    | fused_topk | `4096-128-8-softmax-False` | 0.0117 | 0.8 | 7 | 2% |
