#!/usr/bin/env python3
"""Generate the Benchmarks page from a nightly bench XML — honest edition.

Design contract (no cheating):
  * Per family: a roofline scatter (log-log) with TileOPs AND baseline points.
    Log axes absorb the huge cross-op scale variance; "good" reads as distance
    to the ceiling and to the nearest competitor, not as absolute bar height.
  * Arithmetic intensity AI = achieved_tflops / achieved_bandwidth_tbs (FLOP and
    bytes cancel latency), available from existing artifacts.
  * "% of roofline" = achieved / min(compute_peak[dtype], AI * HBM_peak). Honest
    for GEMM/elementwise (ceiling is reachable); for attention/fused ops the
    ceiling is intrinsically far above SOTA, so good/bad there is judged vs the
    strongest competitive baseline, never vs the unreachable roof.
  * torch / torch-ref are reference only — never a headline speedup.
  * Every benchmarked op is shown, underperformers included; ops with neither a
    competitive baseline nor a reachable roofline are marked "undetermined".

Usage:
    python scripts/gen_bench_pages.py --bench-xml <xml> [--test-xml <xml>] \
        --commit <sha> --date <YYYY-MM-DD> --gpu "NVIDIA H200"
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# --- H200 SXM dense peaks ---------------------------------------------------
PEAK_BW = 4.8  # TB/s HBM3e
PEAK_TF = {"fp8": 1978.9, "float16": 989.5, "bfloat16": 989.5, "float32": 494.7}
PEAK_TF_DEFAULT = 989.5

# Baselines that count as real competitors (torch/torch-ref are reference only).
COMPETITIVE = {"fa3", "flashinfer", "triton", "triton-tma", "deepgemm",
               "torch-cublas", "vllm", "vllm-triton", "fla", "mamba"}

TILEOPS_COLOR = "#00897b"
BL_STYLE = {
    "fa3": "#ff7043", "flashinfer": "#5c6bc0", "triton": "#ab47bc",
    "triton-tma": "#8e24aa", "deepgemm": "#ec407a", "torch-cublas": "#26a69a",
    "vllm": "#42a5f5", "vllm-triton": "#29b6f6", "fla": "#66bb6a",
    "mamba": "#9ccc65", "torch": "#cfd8dc", "torch-ref": "#cfd8dc",
}
GREEN, YELLOW, RED = "🟢", "🟡", "🔴"

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


def peak_tf(name: str) -> float:
    return PEAK_TF.get(dtype_of(name), PEAK_TF_DEFAULT)


def cfg_ai(c: dict) -> float | None:
    tf, bw = c.get("tileops_tflops"), c.get("tileops_bandwidth_tbs")
    if tf and bw:
        return tf / bw
    return None


def cfg_roofline_pct(c: dict) -> float | None:
    ai = cfg_ai(c)
    tf = c.get("tileops_tflops")
    if ai is None or not tf:
        return None
    attain = min(peak_tf(c["name"]), ai * PEAK_BW)
    return tf / attain * 100 if attain else None


def best_competitor(c: dict):
    """Return (tag, ratio) vs the FASTEST competitive baseline, else (None, None).

    ratio = competitor_latency / tileops_latency  (>1 => TileOPs faster).
    """
    tl = c.get("tileops_latency_ms")
    comp = {t: b for t, b in (c.get("baselines") or {}).items()
            if t in COMPETITIVE and b.get("latency_ms")}
    if not tl or not comp:
        return None, None
    tag = min(comp, key=lambda t: comp[t]["latency_ms"])
    return tag, comp[tag]["latency_ms"] / tl


def verdict(c: dict) -> str:
    """At-a-glance good/bad, honest about what is measurable."""
    tag, ratio = best_competitor(c)
    if ratio is not None:
        mark = GREEN if ratio >= 0.95 else YELLOW if ratio >= 0.8 else RED
        return f"{mark} {ratio:.2f}× {tag}"
    pct = cfg_roofline_pct(c)
    if pct is not None:
        return f"({pct:.0f}% of roof)"
    return "—"


# --- roofline scatter -------------------------------------------------------
def render_roofline(fam: str, ops: dict[str, dict], out_svg: str) -> int:
    pts: dict[str, list] = defaultdict(list)
    dtypes = set()
    for d in ops.values():
        for c in d["configs"]:
            ai = cfg_ai(c)
            tf = c.get("tileops_tflops")
            if ai is None or not tf:
                continue
            dtypes.add(dtype_of(c["name"]))
            pts["TileOPs"].append((ai, tf))
            for tag, bl in (c.get("baselines") or {}).items():
                if bl.get("tflops"):  # same op => same AI
                    pts[tag].append((ai, bl["tflops"]))
    n = len(pts.get("TileOPs", []))
    if not n:
        return 0

    # Zoom each family to its own data window so memory-bound families (all
    # low-AI) fill the chart instead of collapsing into a corner.
    all_x = [x for ps in pts.values() for x, _ in ps]
    all_y = [y for ps in pts.values() for y in (v for _, v in ps)]
    xlo, xhi = min(all_x) * 0.4, max(all_x) * 2.5
    ylo, yhi = max(0.3, min(all_y) * 0.55), max(all_y) * 2.2

    fig, ax = plt.subplots(figsize=(9, 5.8), dpi=140)
    ai_grid = np.logspace(np.log10(xlo), np.log10(xhi), 500)
    compute_roofs = sorted({PEAK_TF.get(dt, PEAK_TF_DEFAULT) for dt in dtypes})
    for pk in compute_roofs:
        roof = np.minimum(pk, ai_grid * PEAK_BW)
        ax.plot(ai_grid, roof, color="#37474f", lw=1.4, zorder=5)
        if pk <= yhi:  # flat part is in view → label it
            ax.text(xhi, pk, f" {pk:.0f} TF", fontsize=7, color="#37474f",
                    va="bottom", ha="right")
    top = max(compute_roofs) if compute_roofs else PEAK_TF_DEFAULT
    ax.fill_between(ai_grid, np.minimum(top, ai_grid * PEAK_BW), 1e4,
                    color="#eceff1", alpha=0.45, zorder=0)
    # Annotate the bandwidth diagonal (the reachable ceiling for memory-bound).
    if (xlo * PEAK_BW) < yhi:
        ax.text(xlo * 1.3, xlo * 1.3 * PEAK_BW, "HBM 4.8 TB/s roof", fontsize=7,
                color="#546e7a", rotation=33, va="bottom")

    # TileOPs last so it sits on top.
    order = [t for t in pts if t != "TileOPs"] + ["TileOPs"]
    for tag in order:
        ps = pts[tag]
        if not ps:
            continue
        xs, ys = zip(*ps)
        if tag == "TileOPs":
            ax.scatter(xs, ys, marker="^", s=66, color=TILEOPS_COLOR,
                       edgecolor="white", lw=0.6, zorder=6, label="TileOPs")
        else:
            ref = tag in ("torch", "torch-ref")
            ax.scatter(xs, ys, marker="o", s=34, color=BL_STYLE.get(tag, "#90a4ae"),
                       alpha=0.55 if ref else 0.85, edgecolor="white", lw=0.4,
                       zorder=3 if ref else 4,
                       label=f"{tag} (ref)" if ref else tag)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("Arithmetic intensity  (FLOP / byte)", fontsize=9)
    ax.set_ylabel("Achieved TFLOPS", fontsize=9)
    ax.set_title(f"{FAMILY_TITLE.get(fam, fam)} — roofline (H200)", fontsize=12,
                 fontweight="bold", loc="left")
    ax.grid(True, which="both", color="#f4f4f4", zorder=0)
    ax.legend(loc="lower right", fontsize=7.5, frameon=True, framealpha=0.9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_svg, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return n


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
    ap.add_argument("--date", default="unknown")
    ap.add_argument("--gpu", default="unknown")
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

    assets = os.path.join(REPO, "docs", "benchmarks", "assets")
    os.makedirs(assets, exist_ok=True)

    n_ops = sum(len(v) for v in fams.values())
    n_cfg = sum(len(d["configs"]) for v in fams.values() for d in v.values())
    ordered = [f for f in FAMILY_ORDER if f in fams] + [f for f in fams if f not in FAMILY_ORDER]

    lines = [
        "# Benchmarks", "",
        '!!! info "Nightly performance snapshot"',
        f"    **GPU:** {args.gpu} · **Commit:** "
        f"[`{args.commit}`](https://github.com/tile-ai/TileOPs/commit/{args.commit}) · "
        f"**Date:** {args.date}", "",
        f"    **{n_ops} ops** across **{len(ordered)} families** ({n_cfg} configs). "
        "Each family is shown on its own **roofline** (log–log): the solid line is "
        "the H200 theoretical ceiling; points are achieved throughput. A point "
        "near the ceiling uses the hardware well; a point near a baseline matches "
        "that kernel.", "",
        "!!! warning \"How to read good vs. underperforming\"",
        "    Distance below the ceiling is *not* by itself a verdict — flash-style "
        "attention is intrinsically far below the GEMM ceiling, and SOTA kernels "
        "(FA3, FlashInfer) sit there too. Good/bad is judged **against the "
        "strongest competitive baseline** where one exists "
        f"({GREEN} ≥0.95× · {YELLOW} 0.80–0.95× · {RED} <0.80×); against the "
        "**roofline** only where the ceiling is reachable (GEMM, memory-bound "
        "ops); and left **undetermined** otherwise. `torch` is reference only.", "",
    ]

    for fam in ordered:
        ops = fams[fam]
        svg = os.path.join(assets, f"family_{fam}.svg")
        plotted = render_roofline(fam, ops, svg)
        title = FAMILY_TITLE.get(fam, fam)
        lines += [f"## {title}  <small>({len(ops)} ops)</small>", ""]
        if plotted:
            lines += [f"![{title} roofline](assets/family_{fam}.svg)", ""]
        lines += ['??? note "Per-config detail"', "",
                  "    | Op | Config | Latency (ms) | TFLOPS | AI | % roof | Status |",
                  "    | --- | --- | ---: | ---: | ---: | ---: | --- |"]
        for op in sorted(ops):
            tstat = test_agg.get(op, {})
            badge = (" ✅" if not tstat.get("failed") else " ❌") if tstat else ""
            for c in ops[op]["configs"]:
                cfg = c["name"].split("[")[-1].rstrip("]") if "[" in c["name"] else c["name"]
                lat = c.get("tileops_latency_ms")
                tf = c.get("tileops_tflops")
                ai = cfg_ai(c)
                pct = cfg_roofline_pct(c)
                lines.append(
                    f"    | {op.replace('Op', '')}{badge} | `{cfg}` | "
                    f"{lat if lat is not None else '–'} | "
                    f"{f'{tf:.1f}' if tf else '–'} | "
                    f"{f'{ai:.0f}' if ai else '–'} | "
                    f"{f'{pct:.0f}%' if pct else '–'} | {verdict(c)} |")
        lines.append("")

    out_md = os.path.join(REPO, "docs", "benchmarks", "index.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_md}: {n_ops} ops, {len(ordered)} families, {n_cfg} configs")


if __name__ == "__main__":
    main()
