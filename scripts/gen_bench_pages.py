#!/usr/bin/env python3
"""Generate the Benchmarks page (tables only) from a nightly bench XML.

Design contract (no cheating):
  * One visible summary table per op family (one row per op): config count,
    median TFLOPS, median % of roofline, median speedup vs the strongest
    competitive baseline, and an honest good/bad status.
  * Arithmetic intensity AI = achieved_tflops / achieved_bandwidth_tbs.
  * "% of roofline" = achieved / min(compute_peak[dtype], AI * HBM_peak).
  * Status: judged vs the strongest competitive baseline where one exists
    (torch/torch-ref are reference only, never a headline); else vs the
    roofline only where it is reachable (memory-bound ops); else undetermined.
  * Every benchmarked op is shown, underperformers included.

Usage:
    python scripts/gen_bench_pages.py --bench-xml <xml> [--test-xml <xml>] \
        --commit <sha> --date <YYYY-MM-DD> --gpu "NVIDIA H200"
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# --- H200 SXM dense peaks ---------------------------------------------------
PEAK_BW = 4.8  # TB/s HBM3e
PEAK_TF = {"fp8": 1978.9, "float16": 989.5, "bfloat16": 989.5, "float32": 494.7}
PEAK_TF_DEFAULT = 989.5

COMPETITIVE = {"fa3", "flashinfer", "triton", "triton-tma", "deepgemm",
               "torch-cublas", "vllm", "vllm-triton", "fla", "mamba"}

GREEN, YELLOW, RED, NA = "🟢", "🟡", "🔴", "—"

FAMILY_ORDER = [
    "attention", "linear_attention", "scan", "normalization", "moe",
    "linear_algebra", "reduction", "elementwise", "convolution", "pool",
    "quantization", "positional", "fft", "mhc", "topk", "other",
]
FAMILY_TITLE = {
    "attention": "Attention", "linear_attention": "Linear Attention / SSM",
    "scan": "Scan", "normalization": "Normalization", "moe": "Mixture of Experts",
    "linear_algebra": "Linear Algebra (GEMM)", "reduction": "Reduction",
    "elementwise": "Elementwise", "convolution": "Convolution", "pool": "Pooling",
    "quantization": "Quantization", "positional": "Positional Encoding",
    "fft": "FFT", "mhc": "MHC", "topk": "Top-k", "other": "Other",
}
_KEYWORD_FAMILY = [
    (("mamba", "deltanet", "gla", "linear_attn", "recurrence", "ssd", "ssm",
      "engram"), "linear_attention"),
    (("cumsum", "cumulative", "scan", "cumprod"), "scan"),
    (("layer_norm", "rms_norm", "rmsnorm", "batch_norm", "group_norm",
      "ada_layer", "norm"), "normalization"),
    (("grouped_gemm", "gemm", "matmul", "linear"), "linear_algebra"),
    (("moe", "expert"), "moe"),
    (("conv",), "convolution"), (("pool",), "pool"), (("fft",), "fft"),
    (("quant", "fp8"), "quantization"),
    (("rope", "rotary", "positional"), "positional"),
    (("mhc",), "mhc"), (("topk", "top_k"), "topk"),
    (("attention", "gqa", "mha", "mla", "flash", "kv_cache", "dsa"), "attention"),
    (("reduce", "argmax", "argmin", "argreduce", "mean", "sum", "max", "min"),
     "reduction"),
]
_MODULE_FAMILY = {"attention": "attention", "elementwise": "elementwise",
                  "reduction": "reduction", "norm": "normalization", "moe": "moe"}


def family_of(op: str, op_module: str | None) -> str:
    mod = (op_module or "").lower()
    parts = mod.split(".")
    if len(parts) >= 4 and parts[0] == "tileops" and parts[1] == "ops":
        if parts[2] in _MODULE_FAMILY:
            return _MODULE_FAMILY[parts[2]]
    hay = f"{mod} {op.lower()}"
    for keys, fam in _KEYWORD_FAMILY:
        if any(k in hay for k in keys):
            return fam
    return "elementwise" if not mod or len(parts) <= 3 else "other"


def dtype_of(name: str) -> str:
    n = name.lower()
    for d in ("fp8", "bfloat16", "float16", "float32"):
        if d in n:
            return d
    return "default"


def cfg_ai(c: dict) -> float | None:
    tf, bw = c.get("tileops_tflops"), c.get("tileops_bandwidth_tbs")
    return tf / bw if (tf and bw) else None


def cfg_roofline_pct(c: dict) -> float | None:
    ai, tf = cfg_ai(c), c.get("tileops_tflops")
    if ai is None or not tf:
        return None
    attain = min(PEAK_TF.get(dtype_of(c["name"]), PEAK_TF_DEFAULT), ai * PEAK_BW)
    return tf / attain * 100 if attain else None


def cfg_memory_bound(c: dict) -> bool:
    ai = cfg_ai(c)
    return ai is not None and ai * PEAK_BW < PEAK_TF.get(dtype_of(c["name"]), PEAK_TF_DEFAULT)


def cfg_competitor(c: dict):
    """(tag, ratio) vs the FASTEST competitive baseline; ratio>1 => TileOPs faster."""
    tl = c.get("tileops_latency_ms")
    comp = {t: b for t, b in (c.get("baselines") or {}).items()
            if t in COMPETITIVE and b.get("latency_ms")}
    if not tl or not comp:
        return None, None
    tag = min(comp, key=lambda t: comp[t]["latency_ms"])
    return tag, comp[tag]["latency_ms"] / tl


_GH = "https://github.com/tile-ai/TileOPs"


def op_link(op: str, module: str | None) -> str:
    """Link an op to its source: the module's .py file if it exists, else search."""
    if module and module.startswith("tileops."):
        rel = module.replace(".", "/") + ".py"
        if os.path.exists(os.path.join(REPO, "TileOPs", rel)):
            return f"{_GH}/blob/main/{rel}"
    return f"{_GH}/search?q=repo%3Atile-ai%2FTileOPs+{op}&type=code"


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def op_summary(configs: list[dict]) -> dict:
    """Aggregate an op's configs into one honest summary row."""
    tflops = _med([c.get("tileops_tflops") for c in configs])
    roof = _med([cfg_roofline_pct(c) for c in configs])
    ratios, tags = [], []
    for c in configs:
        tag, r = cfg_competitor(c)
        if r is not None:
            ratios.append(r)
            tags.append(tag)
    has_comp = bool(ratios)
    med_ratio = _med(ratios) if ratios else None
    sota_tag = Counter(tags).most_common(1)[0][0] if tags else None
    mem_bound = sum(cfg_memory_bound(c) for c in configs) > len(configs) / 2

    if has_comp and med_ratio is not None:
        status = GREEN if med_ratio >= 0.95 else YELLOW if med_ratio >= 0.8 else RED
        sota = f"{status} {med_ratio:.2f}× {sota_tag}"
    elif mem_bound and roof is not None:
        status = GREEN if roof >= 70 else YELLOW if roof >= 40 else RED
        sota = f"{status} {roof:.0f}% roof"
    else:
        status = NA
        sota = NA
    return {"configs": len(configs), "tflops": tflops, "roof": roof,
            "status": status, "sota": sota}


def _load_nightly_report():
    path = os.path.join(REPO, "TileOPs", "scripts", "nightly_report.py")
    spec = importlib.util.spec_from_file_location("nightly_report", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    import sys as _sys
    _sys.argv = ["x"]
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-xml", required=True)
    ap.add_argument("--test-xml")
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--date", default="unknown",
                    help="Benchmark run date (when the data was produced)")
    ap.add_argument("--gpu", default="unknown")
    ap.add_argument("--rendered", default=None,
                    help="Page render timestamp, e.g. '2026-06-16 02:00 UTC'")
    args = ap.parse_args()

    nr = _load_nightly_report()
    bench_rows = nr.parse_bench_xml(args.bench_xml)
    agg = nr.aggregate_bench_results(bench_rows)
    module_of = {r["op"]: r.get("op_module") for r in bench_rows}
    test_agg = {}
    if args.test_xml and os.path.exists(args.test_xml):
        test_agg = nr.aggregate_test_results(nr.parse_test_xml(args.test_xml))

    fams: dict[str, dict] = defaultdict(dict)
    for op, data in agg.items():
        fams[family_of(op, module_of.get(op))][op] = data

    n_ops = sum(len(v) for v in fams.values())
    n_cfg = sum(len(d["configs"]) for v in fams.values() for d in v.values())
    ordered = [f for f in FAMILY_ORDER if f in fams] + [f for f in fams if f not in FAMILY_ORDER]

    _nb = "https://github.com/tile-ai/TileOPs/tree/nightly-bench"
    lines = [
        "# Benchmarks", "",
        '!!! info "Nightly performance snapshot"',
        f"    **GPU:** {args.gpu} · **Commit:** "
        f"[`{args.commit}`](https://github.com/tile-ai/TileOPs/commit/{args.commit}) · "
        f"**Benchmark run:** {args.date} · **{n_ops} ops** / {len(ordered)} families / "
        f"{n_cfg} configs",
    ]
    if args.rendered:
        lines += ["", f"    *Page updated {args.rendered} from the "
                  f"[`nightly-bench`]({_nb}) data snapshot.*"]
    lines += [
        "",
        "**Status**", "",
        f"- Against the strongest competitive baseline where one exists: "
        f"{GREEN} ≥0.95× · {YELLOW} 0.80–0.95× · {RED} <0.80×",
        f"- Against the roofline only where it is reachable (memory-bound ops): "
        f"{GREEN} ≥70% · {YELLOW} 40–70% · {RED} <40%",
        f"- `{NA}` when neither applies",
        "- `torch` is reference only",
        "- **Tests**: ✅ correctness test passed · ❌ failed · `–` no test matched",
        "- **% roof** = achieved ÷ H200 theoretical ceiling at the op's "
        "arithmetic intensity",
        "",
    ]

    for fam in ordered:
        ops = fams[fam]
        lines += [f"## {FAMILY_TITLE.get(fam, fam)}  <small>({len(ops)} ops)</small>", "",
                  "| Op | Tests | Configs | TFLOPS | % roof | vs baseline | Status |",
                  "| --- | :---: | ---: | ---: | ---: | --- | :---: |"]
        rows = []
        for op in ops:
            s = op_summary(ops[op]["configs"])
            tstat = test_agg.get(op, {})
            correct = ("✅" if not tstat.get("failed") else "❌") if tstat else "–"
            rows.append((s["status"], op, correct, s))
        # Sort: failing/underperforming first is misleading; sort by name, but
        # keep it scannable — group by status (good → bad → undetermined).
        rank = {GREEN: 0, YELLOW: 1, RED: 2, NA: 3}
        rows.sort(key=lambda r: (rank.get(r[0], 9), r[1]))
        for status, op, correct, s in rows:
            name = f"[{op.replace('Op', '')}]({op_link(op, module_of.get(op))})"
            lines.append(
                f"| {name} | {correct} | {s['configs']} | "
                f"{f'{s['tflops']:.1f}' if s['tflops'] else '–'} | "
                f"{f'{s['roof']:.0f}%' if s['roof'] is not None else '–'} | "
                f"{s['sota']} | {status} |")
        lines.append("")

        # Collapsible per-config detail.
        lines += ['??? note "Per-config detail"', "",
                  "    | Op | Config | Latency (ms) | TFLOPS | AI | % roof |",
                  "    | --- | --- | ---: | ---: | ---: | ---: |"]
        for op in sorted(ops):
            for c in ops[op]["configs"]:
                cfg = c["name"].split("[")[-1].rstrip("]") if "[" in c["name"] else c["name"]
                lat, tf = c.get("tileops_latency_ms"), c.get("tileops_tflops")
                ai, pct = cfg_ai(c), cfg_roofline_pct(c)
                lines.append(
                    f"    | {op.replace('Op', '')} | `{cfg}` | "
                    f"{lat if lat is not None else '–'} | "
                    f"{f'{tf:.1f}' if tf else '–'} | "
                    f"{f'{ai:.0f}' if ai else '–'} | "
                    f"{f'{pct:.0f}%' if pct is not None else '–'} |")
        lines.append("")

    out_md = os.path.join(REPO, "docs", "benchmarks", "index.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_md}: {n_ops} ops, {len(ordered)} families, {n_cfg} configs")


if __name__ == "__main__":
    main()
