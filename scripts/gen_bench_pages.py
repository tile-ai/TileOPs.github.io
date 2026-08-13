#!/usr/bin/env python3
"""Render the Benchmarks section from a nightly benchmark XML snapshot.

Output is one overview page, one page explaining the numbers, and five data
pages grouped by op domain. `hooks.py` puts them into the site nav in that
order.

What a data row must answer, per op:

  * How fast it is in absolute terms — FLOP/s and HBM bandwidth.
  * How much of the machine that is, against the GPU profile, and which of the
    two resources bounds the workload.
  * How that compares to the best other implementation measured on the same
    workload, reported with the *same* metrics rather than as a bare ratio.
  * Whether the numbers can be trusted — correctness status, sample spread,
    kernel count, and an explicit marker wherever an input was not recorded.

Rules this renderer follows:

  * The compared quantity is ``device_busy_ms``: the time the device spent
    executing the call's kernels. A single-kernel call has no gap between
    kernels by construction, so comparing spans would charge a multi-kernel
    implementation for the host's launch latency and credit a fused one for
    nothing it did. Gap and kernel count are reported separately, never folded
    into the comparison.
  * Every baseline present in the data is shown. Baselines are tiered
    (library kernel / PyTorch native op / eager reference), never discarded:
    a tag the tier table does not know is reported as unclassified.
  * Hardware ceilings come from ``src/tileops/perf/profiles/*.yaml``. With no
    profile for the benchmarked GPU, utilisation columns render blank rather
    than against guessed peaks.
  * A metric whose input is missing renders as the empty marker and the op is
    listed as such. No metric is silently substituted.

Usage:
    python scripts/gen_bench_pages.py --bench-xml <xml> [--test-xml <xml>] \
        [--meta <meta.json>] --commit <sha> --date <YYYY-MM-DD> --gpu "NVIDIA H200"
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TILEOPS = os.path.join(REPO, "TileOPs")
_GH = "https://github.com/tile-ai/TileOPs"
_NB = f"{_GH}/tree/nightly-bench"

GREEN, YELLOW, RED, NA = "🟢", "🟡", "🔴", "—"
EMPTY = "·"  # a metric whose input was not recorded
NOISY_SPREAD = 25.0  # above this the median stops summarising the samples

# --- Baseline tiers ---------------------------------------------------------
# A baseline's tier decides how a comparison against it reads, not whether it
# is shown. Unknown tags fall into "lib" and are reported on stderr so a newly
# added baseline gets classified deliberately.
TIER_LIB, TIER_TORCH, TIER_REF = "lib", "torch", "ref"
_TORCH_NATIVE = {"torch", "torch-autograd", "torch-dequantized-matmul"}
_KNOWN_TAGS = _TORCH_NATIVE | {
    "fa3", "flashinfer", "flashinfer-bmm-fp8", "flashinfer-fp8-blockscale-sm90",
    "fla", "mamba", "vllm", "vllm-triton", "triton", "triton-tma", "deepgemm",
    "marlin-fp16", "marlin-fp32", "torch-cublas", "torch-cudnn", "torch-cufft",
    "torch-scaled-mm", "torch-sdpa",
    "torch-ref", "torch-fp32-ref",
}


def tier_of(tag: str) -> str:
    if tag.endswith("-ref") or tag.endswith("_ref"):
        return TIER_REF
    if tag in _TORCH_NATIVE:
        return TIER_TORCH
    return TIER_LIB


# --- Op families and the pages they group into -----------------------------
FAMILY_TITLE = {
    "attention": "Attention", "linear_attention": "Linear Attention / SSM",
    "scan": "Scan", "normalization": "Normalization", "moe": "Mixture of Experts",
    "linear_algebra": "Linear Algebra (GEMM)", "reduction": "Reduction",
    "elementwise": "Elementwise", "convolution": "Convolution", "pool": "Pooling",
    "quantization": "Quantization", "positional": "Positional Encoding",
    "fft": "FFT", "mhc": "MHC", "topk": "Top-k", "other": "Other",
}
# (slug, page title, families in display order)
DATA_PAGES = [
    ("attention", "Attention", ["attention"]),
    ("linear-attention", "Linear Attention & SSM", ["linear_attention", "scan"]),
    ("gemm-moe", "GEMM, MoE & Quantization",
     ["linear_algebra", "moe", "quantization"]),
    ("elementwise-reduction", "Elementwise & Reduction",
     ["elementwise", "reduction"]),
    ("norm-conv-pool", "Norm, Conv, Pool & Other",
     ["normalization", "convolution", "pool", "positional", "fft", "mhc",
      "topk", "other"]),
]
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


def page_of_family(fam: str) -> str:
    for slug, _, fams in DATA_PAGES:
        if fam in fams:
            return slug
    return DATA_PAGES[-1][0]


# --- GPU profile -----------------------------------------------------------
# Utilisation is reported against the *attainable* ceiling (theoretical peak x
# measured calibration) because that is the number a kernel can actually reach;
# both figures are printed on the overview page.
_PROFILE_FILE = {"h200": ("nvidia h200",), "h20_3e": ("nvidia h20-3e", "h20")}
_DTYPE_PEAK_KEY = {"fp8": "fp8", "bfloat16": "bf16", "bf16": "bf16",
                   "float16": "fp16", "fp16": "fp16", "float32": "tf32"}
_DTYPE_TOKENS = ("fp8", "bfloat16", "bf16", "float16", "fp16", "float32")


def load_gpu_profile(gpu: str) -> dict | None:
    """Load the profile matching a GPU name, or None when none matches."""
    try:
        import yaml
    except ImportError:
        print("warning: pyyaml missing; utilisation columns will be blank",
              file=sys.stderr)
        return None
    prof_dir = os.path.join(TILEOPS, "src", "tileops", "perf", "profiles")
    if not os.path.isdir(prof_dir):
        return None
    want = (gpu or "").strip().lower()
    for stem, aliases in _PROFILE_FILE.items():
        if not any(a in want for a in aliases):
            continue
        path = os.path.join(prof_dir, f"{stem}.yaml")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            return _resolve_profile(yaml.safe_load(f))
    print(f"warning: no GPU profile matches {gpu!r}", file=sys.stderr)
    return None


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _resolve_profile(raw: dict) -> dict:
    """Reduce a profile YAML to {'bw': (theo, attainable), 'tf': {key: (...)}}."""
    def pair(section):
        theo = _num(section.get("theoretical"))
        cal = _num(section.get("calibration")) or 1.0
        return (theo, theo * cal if theo else None)

    tf = {k: pair(v) for k, v in (raw.get("tensor_core") or {}).items()
          if isinstance(v, dict)}
    return {"gpu": raw.get("gpu"), "bw": pair(raw.get("hbm") or {}), "tf": tf}


def dtype_of(config_name: str) -> str | None:
    """The dtype token a workload name carries, if any."""
    n = config_name.lower()
    for tok in _DTYPE_TOKENS:
        if tok in n:
            return tok
    return None


class Machine:
    """The ceilings a workload's utilisation is measured against."""

    def __init__(self, profile: dict | None):
        self.profile = profile

    @property
    def known(self) -> bool:
        return bool(self.profile and self.profile["bw"][1])

    def bw_peak(self) -> tuple[float | None, float | None]:
        """Theoretical and attainable HBM bytes/s."""
        return self.profile["bw"] if self.profile else (None, None)

    def tf_peak(self, dtype: str | None) -> tuple[float | None, float | None]:
        return self.tf_peak_by_key(_DTYPE_PEAK_KEY.get(dtype or ""))

    def tf_peak_by_key(self, key: str | None) -> tuple[float | None, float | None]:
        if not self.profile or not key:
            return (None, None)
        return self.profile["tf"].get(key) or (None, None)


# --- XML parsing -----------------------------------------------------------
# Property names are <tag>_<metric>. Parsing is generic over the metric suffix
# so a metric added on the TileOPs side appears here without a code change.
# Longer suffixes come first: device_busy_p10_ms must not match as latency_ms.
_METRIC_SUFFIXES = (
    "device_busy_p10_ms", "device_busy_p90_ms", "device_busy_ms",
    "latency_p10_ms", "latency_p90_ms", "latency_ms", "gap_ms",
    "bandwidth_tbs", "tflops", "ratio", "n_kernels", "n_samples",
    "flops", "bytes", "dtype", "timing", "variant",
)
_NUMERIC_METRICS = {
    "device_busy_ms", "device_busy_p10_ms", "device_busy_p90_ms",
    "latency_ms", "latency_p10_ms", "latency_p90_ms", "gap_ms", "tflops",
    "bandwidth_tbs", "ratio", "flops", "bytes", "n_kernels", "n_samples",
}
# Legacy duplicates of the first baseline's tag-prefixed keys.
_LEGACY = {"baseline_tag", "baseline_device_busy_ms", "baseline_latency_ms",
           "baseline_tflops", "baseline_ratio", "baseline_bandwidth_tbs"}


def parse_bench_xml(path: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (workloads, failures, skips) from a benchmark JUnit XML."""
    workloads, failures, skips = [], [], []
    for tc in ET.parse(path).iter("testcase"):
        props = {p.attrib["name"]: p.attrib.get("value", "")
                 for p in tc.iter("property")}
        name = tc.attrib.get("name", "")
        skipped = tc.find("skipped")
        bad = tc.find("failure") if tc.find("failure") is not None else tc.find("error")
        if skipped is not None:
            skips.append({"name": name, "op": props.get("op"),
                          "message": skipped.attrib.get("message", "")})
            continue
        if bad is not None:
            failures.append({"name": name, "op": props.get("op"),
                             "message": bad.attrib.get("message", "")})
            continue
        if "op" not in props:
            continue

        impls: dict[str, dict] = defaultdict(dict)
        for key, val in props.items():
            if key in ("op", "op_module") or key in _LEGACY:
                continue
            for suf in _METRIC_SUFFIXES:
                if key.endswith("_" + suf):
                    tag = key[: -len(suf) - 1]
                    impls[tag][suf] = _num(val) if suf in _NUMERIC_METRICS else val
                    break
        workloads.append({
            "name": name,
            "config": name.split("[")[-1].rstrip("]") if "[" in name else name,
            "op": props["op"],
            "op_module": props.get("op_module"),
            "impls": dict(impls),
        })
    return workloads, failures, skips


def parse_test_xml(path: str) -> dict[str, dict]:
    """Per-op correctness tally and worst absolute error."""
    ops: dict[str, dict] = defaultdict(
        lambda: {"passed": 0, "failed": 0, "skipped": 0, "max_abs_err": None})
    for tc in ET.parse(path).iter("testcase"):
        props = {p.attrib["name"]: p.attrib.get("value", "")
                 for p in tc.iter("property")}
        op = props.get("op")
        if not op:
            continue
        d = ops[op]
        if tc.find("skipped") is not None:
            d["skipped"] += 1
        elif tc.find("failure") is not None or tc.find("error") is not None:
            d["failed"] += 1
        else:
            d["passed"] += 1
        err = _num(props.get("max_abs_err"))
        if err is not None:
            d["max_abs_err"] = err if d["max_abs_err"] is None else max(
                d["max_abs_err"], err)
    return dict(ops)


# --- Per-workload metrics --------------------------------------------------
# A recorded 0.0 for tflops/bandwidth means the op reported no FLOPs or no
# bytes for the workload; the derived metric is unavailable, not zero.


def _pos(x) -> float | None:
    return x if isinstance(x, (int, float)) and x > 0 else None


def _busy_of(impl: dict) -> float | None:
    """The device-execution time of one implementation on one workload.

    Falls back to the span for a snapshot taken before the two were recorded
    separately, where the only recorded quantity was the span.
    """
    return _pos(impl.get("device_busy_ms")) or _pos(impl.get("latency_ms"))


def workload_metrics(w: dict, mach: Machine) -> dict:
    """Derive every displayed metric for one benchmarked workload."""
    tl = w["impls"].get("tileops", {})
    busy = _busy_of(tl)
    span = _pos(tl.get("latency_ms"))
    tflops = _pos(tl.get("tflops"))
    bw = _pos(tl.get("bandwidth_tbs"))
    dtype = tl.get("dtype") or dtype_of(w["config"])

    # Prefer the recorded roofline inputs for arithmetic intensity; the ratio of
    # the derived rates is algebraically the same quantity and covers snapshots
    # taken before the inputs were recorded.
    flops, nbytes = _pos(tl.get("flops")), _pos(tl.get("bytes"))
    ai = (flops / nbytes) if (flops and nbytes) else (
        (tflops / bw) if (tflops and bw) else None)

    m = {
        "busy_ms": busy, "span_ms": span, "tflops": tflops, "bw_tbs": bw,
        "dtype": dtype, "flops": flops, "bytes": nbytes, "ai": ai,
        "n_kernels": tl.get("n_kernels"), "n_samples": tl.get("n_samples"),
        "variant": tl.get("variant"),
        "gap_pct": (100 * (span - busy) / span) if (span and busy and span >= busy)
                   else None,
        "spread_pct": None,
        "compute_util": None, "bw_util": None, "sol": None, "bound": None,
        "resident": False,
    }
    p10 = _pos(tl.get("device_busy_p10_ms")) or _pos(tl.get("latency_p10_ms"))
    p90 = _pos(tl.get("device_busy_p90_ms")) or _pos(tl.get("latency_p90_ms"))
    if busy and p10 and p90:
        m["spread_pct"] = (p90 - p10) / busy * 100

    bw_theo, bw_att = mach.bw_peak()
    _, tf_att = mach.tf_peak(dtype)
    if bw and bw_att:
        m["bw_util"] = bw * 1e12 / bw_att * 100
        m["resident"] = bw_theo is not None and bw * 1e12 > bw_theo
    if tflops and tf_att:
        m["compute_util"] = tflops * 1e12 / tf_att * 100
    utils = [(u, b) for u, b in ((m["compute_util"], "compute"),
                                 (m["bw_util"], "memory")) if u is not None]
    if utils:
        m["sol"], m["bound"] = max(utils)
    if m["compute_util"] is None or m["bw_util"] is None:
        m["bound"] = None  # naming a bound needs both ceilings resolved

    # Baselines run the same workload, so its FLOP and byte counts apply to them
    # too; their rates follow from their own device-execution time.
    rivals = {}
    for tag, d in w["impls"].items():
        if tag.startswith("tileops"):
            continue
        b_busy = _busy_of(d)
        if not b_busy:
            continue
        b_span = _pos(d.get("latency_ms"))
        b_bw = _pos(d.get("bandwidth_tbs"))
        if b_bw is None and bw and busy:
            b_bw = bw * busy / b_busy
        b_tf = _pos(d.get("tflops"))
        if b_tf is None and tflops and busy:
            b_tf = tflops * busy / b_busy
        # The benchmark computes the ratio before rounding its times for the
        # XML, so prefer it: a sub-microsecond kernel loses several percent to
        # the write precision of the times alone.
        computed = (b_busy / busy) if busy else None
        recorded = _pos(d.get("ratio"))
        rivals[tag] = {
            "tier": tier_of(tag), "busy_ms": b_busy, "span_ms": b_span,
            "tflops": b_tf, "bw_tbs": b_bw,
            "speedup": recorded or computed,
            "computed_ratio": computed, "recorded_ratio": recorded,
            "gap_pct": (100 * (b_span - b_busy) / b_span)
                       if (b_span and b_span >= b_busy) else None,
        }
    m["rivals"] = rivals
    return m


def best_rival(metrics: list[dict], tiers: tuple[str, ...]):
    """Fastest rival within `tiers` across an op's workloads, and the aggregate
    speedup against that one rival.

    Ratios aggregate by geometric mean, matching how TileOPs PR bodies report a
    speedup across workloads: an arithmetic mean of ratios would let one large
    win outweigh several losses of the same magnitude.
    """
    per_workload = []
    for m in metrics:
        cands = {t: r for t, r in m["rivals"].items()
                 if r["tier"] in tiers and r["speedup"]}
        if cands:
            tag = min(cands, key=lambda t: cands[t]["busy_ms"])
            per_workload.append((tag, cands[tag]["speedup"]))
    if not per_workload:
        return None, None
    tag = Counter(t for t, _ in per_workload).most_common(1)[0][0]
    # The speedup belongs to the named rival only, never a mix of rivals.
    return tag, statistics.geometric_mean([s for t, s in per_workload if t == tag])


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def op_summary(metrics: list[dict]) -> dict:
    """Aggregate an op's workloads into one row.

    Times and rates take the median over the op's workloads, each column
    independently: the shapes and dtypes differ, so the row gives the scale of
    the op, not a workload anyone can reproduce. Ratios take the geometric mean
    against one named rival, and the worst workload's ratio sits beside it so a
    uniform win is distinguishable from a win averaged with a loss.
    """
    s = {
        "workloads": len(metrics),
        "busy_ms": _med([m["busy_ms"] for m in metrics]),
        "tflops": _med([m["tflops"] for m in metrics]),
        "bw_tbs": _med([m["bw_tbs"] for m in metrics]),
        "compute_util": _med([m["compute_util"] for m in metrics]),
        "bw_util": _med([m["bw_util"] for m in metrics]),
        "sol": _med([m["sol"] for m in metrics]),
        "n_kernels": _med([m["n_kernels"] for m in metrics]),
        "gap_pct": _med([m["gap_pct"] for m in metrics]),
        "resident": any(m["resident"] for m in metrics),
    }
    bounds = Counter(m["bound"] for m in metrics if m["bound"])
    s["bound"] = bounds.most_common(1)[0][0] if bounds else None

    tag, ratio = best_rival(metrics, (TIER_LIB, TIER_TORCH))
    ref_only = False
    if tag is None:
        tag, ratio = best_rival(metrics, (TIER_REF,))
        ref_only = tag is not None
    s.update(rival=tag, speedup=ratio, rival_ref_only=ref_only)
    if tag:
        rs = [m["rivals"][tag] for m in metrics if tag in m["rivals"]]
        s["rival_tflops"] = _med([r["tflops"] for r in rs])
        s["rival_bw_tbs"] = _med([r["bw_tbs"] for r in rs])
        s["rival_gap_pct"] = _med([r["gap_pct"] for r in rs])
        s["worst_speedup"] = min([r["speedup"] for r in rs if r["speedup"]],
                                 default=None)
        s["rival_tier"] = tier_of(tag)
    else:
        s.update(rival_tflops=None, rival_bw_tbs=None, rival_gap_pct=None,
                 worst_speedup=None, rival_tier=None)

    # Status: judged against a real alternative where one was measured; against
    # the attainable ceiling otherwise; undetermined when neither input exists.
    if ratio is not None and not ref_only:
        s["status"] = GREEN if ratio >= 0.95 else YELLOW if ratio >= 0.80 else RED
        s["basis"] = "baseline"
    elif s["sol"] is not None:
        s["status"] = GREEN if s["sol"] >= 70 else YELLOW if s["sol"] >= 40 else RED
        s["basis"] = "sol"
    else:
        s["status"] = NA
        s["basis"] = "none"
    return s


# --- Markdown helpers ------------------------------------------------------


def _md(s) -> str:
    """Neutralise a value for a Markdown table cell."""
    out = str(s).replace("\\", "\\\\")
    for ch in ("|", "`", "[", "]", "<", ">"):
        out = out.replace(ch, "\\" + ch)
    return " ".join(out.split())


def _md_code(s) -> str:
    return " ".join(str(s).replace("`", "'").replace("|", "\\|").split())


def _f(x, spec=".1f", suffix="") -> str:
    return f"{format(x, spec)}{suffix}" if x is not None else EMPTY


def _sig(x) -> str:
    """Three significant digits. Throughputs on these pages span six orders of
    magnitude, so a fixed number of decimals would print small ones as zero."""
    if x is None:
        return EMPTY
    if x == 0:
        return "0"
    return f"{x:.3g}" if abs(x) < 1000 else f"{x:,.0f}"


def _pct(x) -> str:
    if x is None:
        return EMPTY
    return f"{x:.0f}%" if x >= 10 else f"{x:.2g}%"


def _speed(x) -> str:
    if x is None:
        return EMPTY
    return f"{x:.2f}×" if x < 100 else f"{x:,.0f}×"


def _test_mark(t: dict | None) -> str:
    if not t:
        return EMPTY
    if t.get("failed"):
        return "❌"
    if t.get("passed"):
        return "✅"
    return "⏭️"


def op_link(op: str, module: str | None, ref: str) -> str:
    if module and module.startswith("tileops."):
        rel = "src/" + module.replace(".", "/") + ".py"
        if os.path.exists(os.path.join(TILEOPS, rel)):
            return f"{_GH}/blob/{ref}/{rel}"
    return f"{_GH}/search?q=repo%3Atile-ai%2FTileOPs+{op}&type=code"


def _op_cell(op: str, module: str | None, ref: str) -> str:
    return f"[{_md(op.removesuffix('Op'))}]({op_link(op, module, ref)})"


def _bound_cell(bound: str | None, resident: bool) -> str:
    cell = bound or EMPTY
    return cell + " ᶜ" if resident else cell


def _rival_cell(tag: str | None, tier: str | None) -> str:
    if not tag:
        return EMPTY
    badge = "" if tier == TIER_LIB else f" _{tier}_"
    return f"`{_md_code(tag)}`{badge}"


# --- Data tables -----------------------------------------------------------

# The summary answers two questions only: how fast, and how it compares. The
# per-workload table below it carries everything else.
SUMMARY_HEADER = (
    "| Op | Test | Workloads | Busy ms | TFLOP/s | HBM TB/s | SOL "
    "| Best alternative | its busy ÷ ours | worst | its TFLOP/s |",
    "| --- | :-: | -: | -: | -: | -: | -: | --- | -: | -: | -: |",
)


def summary_row(op: str, module: str | None, s: dict, tmark: str, ref: str) -> str:
    return (
        f"| {s['status']} {_op_cell(op, module, ref)} | {tmark} "
        f"| {s['workloads']} | {_f(s['busy_ms'], '.4f')} "
        f"| {_sig(s['tflops'])} | {_sig(s['bw_tbs'])} | {_pct(s['sol'])} "
        f"| {_rival_cell(s['rival'], s['rival_tier'])} | {_speed(s['speedup'])} "
        f"| {_speed(s['worst_speedup'])} | {_sig(s['rival_tflops'])} |"
    )


DETAIL_HEADER = (
    "| Workload | dtype | Busy ms | spread | Kernels | gap | TFLOP/s "
    "| HBM TB/s | AI | SOL | Bound | Alternatives (busy ms · its busy ÷ ours) |",
    "| --- | :-: | -: | -: | -: | -: | -: | -: | -: | -: | :-: | --- |",
)


def _workload_label(config: str, dtype: str | None) -> str:
    """The workload name without the dtype it already has its own column for."""
    if dtype and config.endswith("-" + dtype):
        return config[: -len(dtype) - 1] or config
    return config


def detail_row(w: dict, m: dict) -> str:
    rivals = " · ".join(
        f"`{_md_code(t)}` {r['busy_ms']:.4f} ({_speed(r['speedup'])})"
        for t, r in sorted(m["rivals"].items(), key=lambda kv: kv[1]["busy_ms"])
    ) or EMPTY
    spread = _pct(m["spread_pct"])
    if m["spread_pct"] is not None and m["spread_pct"] > NOISY_SPREAD:
        spread += " ⚠"
    return (
        f"| `{_md_code(_workload_label(w['config'], m['dtype']))}` "
        f"| {m['dtype'] or EMPTY} | {_f(m['busy_ms'], '.4f')} | {spread} "
        f"| {_f(m['n_kernels'], '.0f')} | {_pct(m['gap_pct'])} "
        f"| {_sig(m['tflops'])} | {_sig(m['bw_tbs'])} "
        f"| {_f(m['ai'], '.0f')} | {_pct(m['sol'])} "
        f"| {_bound_cell(m['bound'], m['resident'])} | {rivals} |"
    )


# --- Pages -----------------------------------------------------------------

# Environment keys in the order the overview table shows them; anything else
# recorded in meta.json is appended so a newly published fact is never dropped.
ENV_ORDER = ["image", "digest", "gpu", "driver", "cuda", "torch", "tilelang",
             "flashinfer", "flash-attn", "fa3", "vllm", "triton", "mamba-ssm",
             "deep_gemm", "fla", "timer", "warmup_ms", "repeat_ms"]


def env_block(meta: dict, timing: str | None) -> list[str]:
    """The stack the numbers were produced on, from the published meta.json."""
    env = dict(meta.get("environment") or {})
    if timing and "timer" not in env:
        env["timer"] = timing
    lines = ["## Environment", ""]
    if not env:
        return lines + [
            '!!! warning "The run did not publish its environment"', "",
            "    The snapshot carries no image, driver, CUDA, torch or library "
            "versions, so a number on these pages cannot be tied to the stack "
            "that produced it. The nightly's publish step fills this in "
            f"through [`meta.json`]({_NB}).", "",
        ]
    keys = [k for k in ENV_ORDER if k in env] + [k for k in env if k not in ENV_ORDER]
    lines += ["| | |", "| --- | --- |"]
    for k in keys:
        lines.append(f"| {_md(k)} | `{_md_code(env[k])}` |")
    missing = [k for k in ("image", "digest", "driver", "cuda", "torch",
                           "tilelang") if k not in env]
    if missing:
        lines += ["", "Not published by this run: "
                  + ", ".join(f"`{k}`" for k in missing) + "."]
    return lines + [""]


def method_block(meta: dict) -> list[str]:
    """How the numbers were taken. Fixed policy of the benchmark layer."""
    env = meta.get("environment") or {}
    warm = env.get("warmup_ms")
    rep = env.get("repeat_ms")
    budget = (f"**{warm} ms warmup, {rep} ms measurement** per implementation"
              if warm and rep else
              "**A fixed warmup and measurement budget** per implementation")
    return [
        "## Method", "",
        "How a row was produced:", "",
        "- **One process, common inputs.** Every implementation of an op is "
        "timed on the same tensors in the same process.",
        f"- {budget}, reported as the median over the samples it fits in.",
        "- **Forward then reversed order.** Timing each implementation twice in "
        "opposite orders keeps drift across the case from landing on whichever "
        "ran last.",
        "- **L2 cleared between iterations.**",
        "- **Compilation and workspace setup excluded.**",
        "- **CUPTI, fail-closed.** A run that cannot collect device activity "
        "fails rather than falling back to a different clock.",
        "",
        "What is compared:", "",
        "- **Device-busy time** — the union of the intervals the device spent "
        "executing that call's kernels.",
        "- **It excludes the gaps between kernels within one call.** The "
        "records cannot separate such a gap into the implementation's own "
        "dependencies and the host being late with the next launch.",
        "- **A single-kernel call has no gap at all**, by construction. "
        "Comparing wall-clock spans would therefore charge a multi-kernel "
        "implementation for launch latency it does not own, and credit a fused "
        "one for nothing it did.",
        "- **Launch structure is reported, not compared.** `gap` and `Kernels` "
        "are their own columns on every data page.",
        "",
    ]


def ceilings_block(mach: Machine, gpu: str) -> list[str]:
    bw_theo, bw_att = mach.bw_peak()
    if not mach.known:
        return [
            "## Hardware ceilings", "",
            f'!!! warning "No GPU profile for {gpu}"', "",
            "    Utilisation, SOL and bound columns are blank: this renderer "
            "does not guess peaks. Add a profile under "
            f"[`src/tileops/perf/profiles/`]({_GH}/tree/main/src/tileops/perf/profiles) "
            "to fill them.", "",
        ]
    lines = [
        "## Hardware ceilings", "",
        "Utilisation is reported against the **attainable** ceiling — the "
        "spec-sheet peak scaled by what "
        f"[`benchmarks/hardware/`]({_GH}/tree/main/benchmarks/hardware) measures "
        "on this GPU. 100% means saturating the machine as microbenchmarks find "
        "it, not as the datasheet advertises it.", "",
        "| Resource | Spec-sheet peak | Attainable | Ratio |",
        "| --- | -: | -: | -: |",
        f"| HBM bandwidth | {bw_theo / 1e12:.2f} TB/s | {bw_att / 1e12:.2f} TB/s "
        f"| {bw_att / bw_theo:.0%} |",
    ]
    for key in ("fp8", "fp16", "bf16", "tf32"):
        theo, att = mach.tf_peak_by_key(key)
        if theo:
            lines.append(f"| Tensor core {key} | {theo / 1e12:.0f} TFLOP/s "
                         f"| {att / 1e12:.0f} TFLOP/s | {att / theo:.0%} |")
    lines += ["", "Source: [`src/tileops/perf/profiles/`]"
              f"({_GH}/tree/main/src/tileops/perf/profiles)", ""]
    return lines


def index_page(args, meta: dict, mach: Machine, rows: list[tuple],
               by_page: dict, spreads: list[float], timing: str | None,
               n_workloads: int, n_failed: int, n_skipped: int) -> str:
    run_id = meta.get("run_id")
    head = [
        "# Benchmarks", "",
        '!!! info "Nightly snapshot"', "",
        f"    **GPU** {args.gpu} · **commit** "
        f"[`{args.commit[:12]}`]({_GH}/commit/{args.commit}) · "
        f"**run date** {args.date} · **{len(rows)} ops**, {n_workloads} workloads",
    ]
    if run_id:
        head.append(f"    · [nightly run]({_GH}/actions/runs/{run_id})")
    if args.rendered:
        head += ["", f"    Page rendered {args.rendered} from the "
                 f"[`nightly-bench`]({_NB}) snapshot."]
    head.append("")

    lines = head + env_block(meta, timing) + method_block(meta) \
        + ceilings_block(mach, args.gpu)

    # Status roll-up.
    by = Counter((s["status"], s["basis"]) for _, _, s, _, _ in rows)
    lines += ["## Where the library stands", "",
              "| | Judged against an alternative | Judged against the ceiling "
              "| Total |", "| --- | -: | -: | -: |"]
    for status, label in ((GREEN, f"{GREEN} at or ahead / ≥70% of SOL"),
                          (YELLOW, f"{YELLOW} 0.80–0.95× / 40–70% of SOL"),
                          (RED, f"{RED} below 0.80× / <40% of SOL"),
                          (NA, f"{NA} no alternative and no ceiling resolved")):
        b, c = by[(status, "baseline")], by[(status, "sol")]
        n = b + c + by[(status, "none")]
        if n:
            lines.append(f"| {label} | {b or EMPTY} | {c or EMPTY} | {n} |")
    n_base = sum(v for (_, basis), v in by.items() if basis == "baseline")
    lines += ["", f"{n_base} of {len(rows)} ops have a non-reference alternative "
              "measured on the identical workload; the rest are judged against "
              "the hardware ceiling only, which is a weaker claim. "
              "[How to read a row](reading.md)", ""]
    if spreads:
        noisy = sum(1 for x in spreads if x > NOISY_SPREAD)
        lines += [f"Measurement noise: median p10→p90 spread "
                  f"{statistics.median(spreads):.1f}% of device-busy time; "
                  f"{noisy} of {len(spreads)} workloads exceed "
                  f"{NOISY_SPREAD:.0f}% and carry `⚠`.", ""]
    if n_failed or n_skipped:
        lines += [f"Absent from every table: {n_failed} workloads errored and "
                  f"{n_skipped} were skipped in this run.", ""]

    # Entry table into the data pages.
    lines += ["## Data", "",
              "| Page | Ops | Workloads | median SOL | " + GREEN + " | "
              + YELLOW + " | " + RED + " | " + NA + " |",
              "| --- | -: | -: | -: | -: | -: | -: | -: |"]
    for slug, title, _ in DATA_PAGES:
        page_rows = by_page.get(slug, [])
        if not page_rows:
            continue
        st = Counter(s["status"] for _, _, s, _, _ in page_rows)
        sols = [s["sol"] for _, _, s, _, _ in page_rows if s["sol"] is not None]
        n_w = sum(s["workloads"] for _, _, s, _, _ in page_rows)
        lines.append(
            f"| [{title}]({slug}.md) | {len(page_rows)} | {n_w} "
            f"| {_pct(statistics.median(sols)) if sols else EMPTY} "
            f"| {st[GREEN] or EMPTY} | {st[YELLOW] or EMPTY} "
            f"| {st[RED] or EMPTY} | {st[NA] or EMPTY} |")
    return "\n".join(lines) + "\n"


def reading_page() -> str:
    lines = [
        "# How to read these numbers", "",
        "Each op family gets two tables. The first is one row per op — how fast "
        "it is and how that compares — with the numbers as medians over the "
        "op's workloads. The second lists every workload behind it, with the "
        "full metric set and every alternative measured on it.", "",
        "## Columns", "",
        "| Column | Meaning |",
        "| --- | --- |",
        "| **Busy ms** | Device execution time: union of the call's kernel "
        "intervals. Every comparison uses it. |",
        "| **gap** | Share of the call's span with no kernel running. Reported, "
        "never compared. |",
        "| **Kernels** | Kernels the call launched. |",
        "| **TFLOP/s** | Required FLOPs ÷ busy. Count from the manifest "
        "`roofline` formula, not a hardware counter. |",
        "| **HBM TB/s** | Required bytes ÷ busy, same source. |",
        "| **SOL** | Achieved ÷ attainable ceiling, whichever of the two binds: "
        "the dtype's FLOP/s or HBM bandwidth. |",
        "| **Bound** | Which resource that is. `ᶜ` = bandwidth above the HBM "
        "peak, i.e. cache-resident. |",
        "| **Best alternative** | Fastest other implementation on the same "
        f"workload. Unlabelled = tuned library kernel, _{TIER_TORCH}_ = PyTorch "
        f"native op, _{TIER_REF}_ = eager composition (a weak bar). |",
        "| **its busy ÷ ours** | The alternative's busy time divided by "
        "ours, aggregated over the op's workloads. >1 = TileOPs faster. |",
        "| **worst** | The same ratio on the op's worst workload. |",
        "| **its TFLOP/s** | The alternative's own rate, same definition. |",
        "| **Test** | ✅ passed · ❌ failed · ⏭️ all skipped · "
        f"`{EMPTY}` no test matched. |",
        "| **spread** | (p90 − p10) ÷ median busy. "
        f"`⚠` above {NOISY_SPREAD:.0f}%. |",
        "| **AI** | Arithmetic intensity: FLOPs ÷ bytes. |",
        "",
        "## How a per-op row is aggregated", "",
        "An op is benchmarked on several shapes and dtypes, so every number in "
        "the one-row-per-op table is an aggregate over its workloads:", "",
        "- **Busy ms, TFLOP/s, HBM TB/s, SOL** — the median over the op's "
        "workloads, each column taken independently. Shapes and dtypes are "
        "mixed, so the row gives the op's scale, not a workload you can "
        "reproduce. The per-workload table is where a single number lives.",
        "- **Best alternative** — per workload, the fastest non-reference "
        "alternative is picked; the op is then labelled with whichever "
        "alternative won most often. Only that one alternative's ratios are "
        "aggregated, so the name and the number always belong together.",
        "- **its busy ÷ ours** — the geometric mean of those ratios, the same "
        "statistic "
        "TileOPs PR bodies use. An arithmetic mean would let one large win "
        "outweigh several losses of equal size.",
        "- **worst** — the smallest ratio among them, not an average.",
        "- **its TFLOP/s** — the median of that alternative's rate over the "
        "workloads where it ran, which need not be all of them.",
        "- **Workloads** — how many workloads the aggregate covers.",
        "",
        "The overview page adds one more layer: its `median SOL` per page is a "
        "median of those per-op medians.",
        "",
        "## Status", "",
        "The dot on an op name is judged against the best non-reference "
        f"alternative where one was measured ({GREEN} ≥0.95× · {YELLOW} "
        f"0.80–0.95× · {RED} <0.80×), otherwise against SOL ({GREEN} ≥70% · "
        f"{YELLOW} 40–70% · {RED} <40%), otherwise `{NA}`.",
        "",
        "A comparison against an alternative is the stronger claim: the ceiling "
        "says how much of the machine a kernel uses, not whether someone else's "
        "kernel does the same work faster.",
        "",
        "## Empty cells", "",
        f"`{EMPTY}` means an input to that metric was not recorded, never that "
        "the value is zero: the op reported no FLOP or byte count for that "
        "workload, no alternative ran on it, or its dtype ceiling could not be "
        "resolved.",
        "",
        "## Why device-busy time", "",
        "The wall-clock span of a call includes the gaps between its kernels. "
        "Those gaps are dominated by how fast the host issues the next launch, "
        "which is a property of the benchmark loop rather than of the kernel: "
        "measured on this suite they are a roughly constant per-call cost, so "
        "the same implementation appears to gain or lose against a rival purely "
        "with problem size. A call that launches one fused kernel has no gap at "
        "all. Comparing spans therefore rewards fusion twice and penalises "
        "multi-kernel implementations for the host. Device-busy time excludes "
        "the gaps on both sides; the gap and kernel-count columns keep the "
        "launch structure visible on its own.",
        "",
    ]
    return "\n".join(lines) + "\n"


def data_page(title: str, fams: list[str], rows_by_fam: dict,
              metrics_by_op: dict, workloads_of: dict, ref: str) -> str:
    rank = {GREEN: 0, YELLOW: 1, RED: 2, NA: 3}
    lines = [f"# {title}", "",
             "One row per op, then every workload behind it. Column meanings "
             "are on [How to read these numbers](reading.md).", ""]
    for fam in fams:
        rows = rows_by_fam.get(fam)
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: (rank.get(r[2]["status"], 9), r[0]))
        n_w = sum(s["workloads"] for _, _, s, _, _ in rows)
        # The wrapper is a styling hook: extra.css keeps these dense numeric
        # cells on one line and lets the table scroll instead of wrapping.
        lines += [f"## {FAMILY_TITLE.get(fam, fam)} "
                  f"<small>({len(rows)} ops, {n_w} workloads)</small>", "",
                  '<div class="datatable" markdown="1">', "", *SUMMARY_HEADER]
        for op, module, s, tmark, _ in rows:
            lines.append(summary_row(op, module, s, tmark, ref))
        # One table per op rather than one per family: the op name would
        # otherwise repeat down the widest column of every row.
        lines += ["", "</div>", ""]
        for op, module, s, _, _ in rows:
            lines += [f"### {s['status']} {_md(op.removesuffix('Op'))} "
                      f"<small>({s['workloads']} workloads)</small>", "",
                      '<div class="datatable" markdown="1">', "", *DETAIL_HEADER]
            for w, m in sorted(zip(workloads_of[op], metrics_by_op[op]),
                               key=lambda z: z[0]["config"]):
                lines.append(detail_row(w, m))
            lines += ["", "</div>", ""]
    return "\n".join(lines) + "\n"


# --- Main ------------------------------------------------------------------


RATIO_DRIFT = 0.10


def collect_ratio_drift(workloads: list[dict], metrics: list[dict]) -> list[tuple]:
    """Workloads where the ratio the benchmark recorded and the one recomputed
    from the times it published disagree by more than `RATIO_DRIFT`.

    Small disagreements are the write precision of the times; a large one means
    the two layers are no longer comparing the same quantity, which is how this
    page last went wrong.
    """
    drift = []
    for w, m in zip(workloads, metrics):
        for tag, r in m["rivals"].items():
            rec, comp = r["recorded_ratio"], r["computed_ratio"]
            if rec and comp and abs(rec - comp) / rec > RATIO_DRIFT:
                drift.append((w["name"], tag, rec, comp))
    return drift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-xml", required=True)
    ap.add_argument("--test-xml")
    ap.add_argument("--meta", help="published meta.json (environment, run id)")
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--date", default="unknown")
    ap.add_argument("--gpu", default="unknown")
    ap.add_argument("--rendered", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    ref = args.commit if args.commit and args.commit != "unknown" else "main"

    meta = {}
    if args.meta and os.path.exists(args.meta):
        with open(args.meta, encoding="utf-8") as f:
            meta = json.load(f)

    workloads, failures, skips = parse_bench_xml(args.bench_xml)
    tests = (parse_test_xml(args.test_xml)
             if args.test_xml and os.path.exists(args.test_xml) else {})
    mach = Machine(load_gpu_profile(args.gpu))

    unclassified = sorted({
        t for w in workloads for t in w["impls"]
        if not t.startswith("tileops") and t not in _KNOWN_TAGS})
    timings = Counter(w["impls"].get("tileops", {}).get("timing")
                      for w in workloads).most_common(1)
    timing = timings[0][0] if timings else None

    metrics_by_op: dict[str, list[dict]] = defaultdict(list)
    workloads_of: dict[str, list[dict]] = defaultdict(list)
    module_of: dict[str, str | None] = {}
    for w in workloads:
        metrics_by_op[w["op"]].append(workload_metrics(w, mach))
        workloads_of[w["op"]].append(w)
        module_of.setdefault(w["op"], w["op_module"])

    rows_by_fam: dict[str, list[tuple]] = defaultdict(list)
    by_page: dict[str, list[tuple]] = defaultdict(list)
    all_rows = []
    for op, ms in metrics_by_op.items():
        s = op_summary(ms)
        row = (op, module_of.get(op), s, _test_mark(tests.get(op)), ref)
        fam = family_of(op, module_of.get(op))
        rows_by_fam[fam].append(row)
        by_page[page_of_family(fam)].append(row)
        all_rows.append(row)

    ratio_drift = []
    for op in metrics_by_op:
        ratio_drift += collect_ratio_drift(workloads_of[op], metrics_by_op[op])
    spreads = [m["spread_pct"] for ms in metrics_by_op.values() for m in ms
               if m["spread_pct"] is not None]

    out_dir = args.out_dir or os.path.join(REPO, "docs", "benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    pages = {
        "index.md": index_page(args, meta, mach, all_rows, by_page, spreads,
                               timing, len(workloads), len(failures),
                               len(skips)),
        "reading.md": reading_page(),
    }
    for slug, title, fams in DATA_PAGES:
        if any(rows_by_fam.get(f) for f in fams):
            pages[f"{slug}.md"] = data_page(title, fams, rows_by_fam,
                                            metrics_by_op, workloads_of, ref)
    for name, text in pages.items():
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            f.write(text)

    print(f"wrote {len(pages)} pages to {out_dir}: {len(all_rows)} ops, "
          f"{len(workloads)} workloads, {len(failures)} failed, "
          f"{len(skips)} skipped")
    if ratio_drift:
        print(f"warning: {len(ratio_drift)} workloads where the recorded ratio "
              "disagrees with the computed one", file=sys.stderr)
    if unclassified:
        print("warning: baseline tags with no tier: "
              + ", ".join(unclassified), file=sys.stderr)


if __name__ == "__main__":
    main()
