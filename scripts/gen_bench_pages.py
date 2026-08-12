#!/usr/bin/env python3
"""Render the Benchmarks page from a nightly benchmark XML snapshot.

What the page must answer, per op:

  * How fast is it in absolute terms — FLOP/s and HBM GB/s.
  * How much of the machine that is — compute utilisation and bandwidth
    utilisation against the GPU profile, and which of the two is the limit.
  * How that compares to the best other implementation available, reported
    with the *same* metrics, never as a bare ratio.
  * Whether the numbers can be trusted — correctness status, sample spread,
    and an explicit marker whenever an input to a metric was not recorded.

Rules this renderer follows:

  * Every baseline present in the data is shown. Baselines are tiered
    (library kernel / PyTorch native op / eager reference), never discarded:
    a tag the tier table does not know is reported as unclassified, not
    dropped.
  * Hardware ceilings come from ``src/tileops/perf/profiles/*.yaml``. With no
    profile for the benchmarked GPU, utilisation columns render blank rather
    than against guessed peaks.
  * A metric whose input is missing renders as the empty marker, and the op
    is listed under Data gaps. No metric is silently substituted.

Usage:
    python scripts/gen_bench_pages.py --bench-xml <xml> [--test-xml <xml>] \
        --commit <sha> --date <YYYY-MM-DD> --gpu "NVIDIA H200"
"""
from __future__ import annotations

import argparse
import os
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TILEOPS = os.path.join(REPO, "TileOPs")
_GH = "https://github.com/tile-ai/TileOPs"
_NB = f"{_GH}/tree/nightly-bench"

GREEN, YELLOW, RED, NA = "🟢", "🟡", "🔴", "—"
EMPTY = "·"  # a metric whose input was not recorded

# --- Baseline tiers ---------------------------------------------------------
# A baseline's tier decides how a comparison against it reads, not whether it
# is shown. Unknown tags fall into "lib" and are listed under Data gaps so a
# newly added baseline gets classified deliberately.
TIER_LIB, TIER_TORCH, TIER_REF = "lib", "torch", "ref"
_TORCH_NATIVE = {"torch", "torch-autograd", "torch-dequantized-matmul"}
TIER_LABEL = {
    TIER_LIB: "tuned library kernel",
    TIER_TORCH: "PyTorch native op",
    TIER_REF: "eager reference composition",
}
# Tags whose tier is settled; anything else is reported as unclassified.
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


# --- Op families -----------------------------------------------------------
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


# --- GPU profile -----------------------------------------------------------
# Utilisation is reported against the *attainable* ceiling (theoretical peak x
# measured calibration) because that is the number a kernel can actually reach;
# both figures are printed in the page header.
_PROFILE_FILE = {"h200": ("nvidia h200",), "h20_3e": ("nvidia h20-3e", "h20")}
# dtype token in a config name -> tensor_core section in the profile
_DTYPE_PEAK_KEY = {"fp8": "fp8", "bfloat16": "bf16", "bf16": "bf16",
                   "float16": "fp16", "fp16": "fp16", "float32": "tf32"}
_DTYPE_TOKENS = ("fp8", "bfloat16", "bf16", "float16", "fp16", "float32")


def load_gpu_profile(gpu: str) -> dict | None:
    """Load the profile matching a GPU name, or None when none matches."""
    try:
        import yaml
    except ImportError:
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
            raw = yaml.safe_load(f)
        return _resolve_profile(raw)
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

    hbm = raw.get("hbm") or {}
    tf = {k: pair(v) for k, v in (raw.get("tensor_core") or {}).items()
          if isinstance(v, dict)}
    return {"gpu": raw.get("gpu"), "bw": pair(hbm), "tf": tf}


def dtype_of(config_name: str) -> str | None:
    """The dtype token a config name carries, if any."""
    n = config_name.lower()
    for tok in _DTYPE_TOKENS:
        if tok in n:
            return tok
    return None


# --- XML parsing -----------------------------------------------------------
# Property names are <tag>_<metric>. Parsing is generic over the metric suffix
# so a metric added on the TileOPs side appears here without a code change.
_METRIC_SUFFIXES = (
    "latency_p10_ms", "latency_p90_ms", "latency_ms", "bandwidth_tbs",
    "tflops", "ratio", "n_samples", "flops", "bytes", "dtype", "timing",
    "variant",
)
_NUMERIC_METRICS = {"latency_ms", "latency_p10_ms", "latency_p90_ms", "tflops",
                    "bandwidth_tbs", "ratio", "flops", "bytes", "n_samples"}
# Legacy duplicates of the first baseline's tag-prefixed keys.
_LEGACY = {"baseline_tag", "baseline_latency_ms", "baseline_tflops",
           "baseline_ratio", "baseline_bandwidth_tbs"}


def parse_bench_xml(path: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (configs, failures, skips) from a benchmark JUnit XML."""
    configs, failures, skips = [], [], []
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
        configs.append({
            "name": name,
            "config": name.split("[")[-1].rstrip("]") if "[" in name else name,
            "op": props["op"],
            "op_module": props.get("op_module"),
            "impls": dict(impls),
        })
    return configs, failures, skips


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
            d["max_abs_err"] = err if d["max_abs_err"] is None else max(d["max_abs_err"], err)
    return dict(ops)


# --- Per-config metrics ----------------------------------------------------
# A recorded 0.0 for tflops/bandwidth means the op reported no FLOPs or no
# bytes for the workload; the derived metric is unavailable, not zero.


def _pos(x) -> float | None:
    return x if isinstance(x, (int, float)) and x > 0 else None


class Machine:
    """The ceilings a config's utilisation is measured against."""

    def __init__(self, profile: dict | None):
        self.profile = profile

    @property
    def known(self) -> bool:
        return bool(self.profile and self.profile["bw"][1])

    def bw_peak(self) -> tuple[float | None, float | None]:
        """Theoretical and attainable HBM bytes/s."""
        return self.profile["bw"] if self.profile else (None, None)

    def tf_peak(self, dtype: str | None) -> tuple[float | None, float | None]:
        """Theoretical and attainable FLOP/s for a dtype token."""
        return self.tf_peak_by_key(_DTYPE_PEAK_KEY.get(dtype or ""))

    def tf_peak_by_key(self, key: str | None) -> tuple[float | None, float | None]:
        """Theoretical and attainable FLOP/s for a profile tensor_core key."""
        if not self.profile or not key:
            return (None, None)
        return self.profile["tf"].get(key) or (None, None)


def config_metrics(cfg: dict, mach: Machine) -> dict:
    """Derive every displayed metric for one benchmarked config."""
    tl = cfg["impls"].get("tileops", {})
    lat = _pos(tl.get("latency_ms"))
    tflops = _pos(tl.get("tflops"))
    bw = _pos(tl.get("bandwidth_tbs"))
    dtype = tl.get("dtype") or dtype_of(cfg["config"])

    # Prefer the recorded roofline inputs for arithmetic intensity; fall back to
    # the ratio of the derived rates, which is algebraically the same quantity,
    # for snapshots taken before the inputs were recorded.
    flops, nbytes = _pos(tl.get("flops")), _pos(tl.get("bytes"))
    ai = (flops / nbytes) if (flops and nbytes) else (
        (tflops / bw) if (tflops and bw) else None)

    m = {
        "latency_ms": lat, "tflops": tflops, "bw_tbs": bw, "dtype": dtype,
        "flops": flops, "bytes": nbytes, "ai": ai,
        "spread_pct": None, "n_samples": tl.get("n_samples"),
        "variant": tl.get("variant"),
        "compute_util": None, "bw_util": None, "sol": None, "bound": None,
        "resident": False,
    }
    p10, p90 = _pos(tl.get("latency_p10_ms")), _pos(tl.get("latency_p90_ms"))
    if lat and p10 and p90:
        m["spread_pct"] = (p90 - p10) / lat * 100

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
    # Bound type is only meaningful when both ceilings were resolvable.
    if m["compute_util"] is None or m["bw_util"] is None:
        m["bound"] = None

    # Baselines: same workload, so bytes and FLOPs match TileOPs and the
    # baseline's bandwidth follows from its latency.
    rivals = {}
    for tag, d in cfg["impls"].items():
        if tag.startswith("tileops"):
            continue
        b_lat = _pos(d.get("latency_ms"))
        if not b_lat:
            continue
        b_bw = _pos(d.get("bandwidth_tbs"))
        if b_bw is None and bw and lat:
            b_bw = bw * lat / b_lat
        b_tf = _pos(d.get("tflops"))
        if b_tf is None and tflops and lat:
            b_tf = tflops * lat / b_lat
        rivals[tag] = {
            "tier": tier_of(tag), "latency_ms": b_lat, "tflops": b_tf,
            "bw_tbs": b_bw,
            "speedup": (b_lat / lat) if lat else None,
            "bw_util": (b_bw * 1e12 / bw_att * 100) if (b_bw and bw_att) else None,
        }
    m["rivals"] = rivals
    return m


def best_rival(metrics: list[dict], tiers: tuple[str, ...]) -> tuple[str | None, float | None]:
    """Fastest rival within `tiers` across an op's configs, and the median
    speedup against it over the configs where it ran."""
    per_cfg = []
    for m in metrics:
        cands = {t: r for t, r in m["rivals"].items()
                 if r["tier"] in tiers and r["speedup"]}
        if cands:
            tag = min(cands, key=lambda t: cands[t]["latency_ms"])
            per_cfg.append((tag, cands[tag]["speedup"]))
    if not per_cfg:
        return None, None
    tag = Counter(t for t, _ in per_cfg).most_common(1)[0][0]
    # Report the speedup against the named rival only, never a mixed median.
    ratios = [s for t, s in per_cfg if t == tag]
    return tag, statistics.median(ratios)


def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def op_summary(metrics: list[dict]) -> dict:
    """Aggregate an op's configs into one row. Medians across configs; the
    worst case is reported next to the speedup so a uniform win is
    distinguishable from an average of a win and a loss."""
    s = {
        "configs": len(metrics),
        "tflops": _med([m["tflops"] for m in metrics]),
        "bw_tbs": _med([m["bw_tbs"] for m in metrics]),
        "compute_util": _med([m["compute_util"] for m in metrics]),
        "bw_util": _med([m["bw_util"] for m in metrics]),
        "sol": _med([m["sol"] for m in metrics]),
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
        s["worst_speedup"] = min([r["speedup"] for r in rs if r["speedup"]],
                                 default=None)
        s["rival_tier"] = tier_of(tag)
    else:
        s.update(rival_tflops=None, rival_bw_tbs=None, worst_speedup=None,
                 rival_tier=None)

    # Status: judged against a real alternative where one was measured;
    # against the attainable ceiling otherwise; undetermined when neither
    # input exists.
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
    """Three significant digits. Throughputs on this page span six orders of
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


# --- Page sections ---------------------------------------------------------

SUMMARY_HEADER = (
    "| Op | Test | Workloads | Bound | TFLOP/s | of peak | HBM TB/s | of peak "
    "| SOL | Best alternative | Speedup | worst | its TFLOP/s | its TB/s |",
    "| --- | :-: | -: | :-: | -: | -: | -: | -: | -: | --- | -: | -: | -: | -: |",
)


def _bound_cell(bound: str | None, resident: bool) -> str:
    cell = bound or EMPTY
    return cell + " ᶜ" if resident else cell


def _rival_cell(tag: str | None, tier: str | None) -> str:
    if not tag:
        return EMPTY
    badge = "" if tier == TIER_LIB else f" _{tier}_"
    return f"`{_md_code(tag)}`{badge}"


def summary_row(op: str, module: str | None, s: dict, tmark: str, ref: str) -> str:
    return (
        f"| {s['status']} {_op_cell(op, module, ref)} | {tmark} | {s['configs']} "
        f"| {_bound_cell(s['bound'], s['resident'])} "
        f"| {_sig(s['tflops'])} | {_pct(s['compute_util'])} "
        f"| {_sig(s['bw_tbs'])} | {_pct(s['bw_util'])} | {_pct(s['sol'])} "
        f"| {_rival_cell(s['rival'], s['rival_tier'])} | {_speed(s['speedup'])} "
        f"| {_speed(s['worst_speedup'])} "
        f"| {_sig(s['rival_tflops'])} | {_sig(s['rival_bw_tbs'])} |"
    )


DETAIL_HEADER = (
    "    | Op | Config | dtype | Latency ms | spread | TFLOP/s | HBM TB/s "
    "| AI | SOL | Bound | Alternatives (latency ms · speedup) |",
    "    | --- | --- | :-: | -: | -: | -: | -: | -: | -: | :-: | --- |",
)
NOISY_SPREAD = 25.0  # above this the median is worth distrusting


def detail_row(op: str, cfg: dict, m: dict) -> str:
    rivals = " · ".join(
        f"`{_md_code(t)}` {r['latency_ms']:.4f} ({_speed(r['speedup'])})"
        for t, r in sorted(m["rivals"].items(), key=lambda kv: kv[1]["latency_ms"])
    ) or EMPTY
    spread = _pct(m["spread_pct"])
    if m["spread_pct"] is not None and m["spread_pct"] > NOISY_SPREAD:
        spread += " ⚠"
    return (
        f"    | {_md(op.removesuffix('Op'))} | `{_md_code(cfg['config'])}` "
        f"| {m['dtype'] or EMPTY} | {_f(m['latency_ms'], '.4f')} "
        f"| {spread} | {_sig(m['tflops'])} | {_sig(m['bw_tbs'])} "
        f"| {_f(m['ai'], '.0f')} | {_pct(m['sol'])} "
        f"| {_bound_cell(m['bound'], m['resident'])} | {rivals} |"
    )


def header_block(args, mach: Machine, n_ops: int, n_fam: int, n_cfg: int,
                 timing: str | None) -> list[str]:
    bw_theo, bw_att = mach.bw_peak()
    lines = [
        "# Benchmarks", "",
        '!!! info "Nightly snapshot"', "",
        f"    **GPU** {args.gpu} · **commit** "
        f"[`{args.commit[:12]}`]({_GH}/commit/{args.commit}) · "
        f"**run date** {args.date} · **{n_ops} ops** across {n_fam} families, "
        f"{n_cfg} workloads",
    ]
    if timing:
        lines.append(f"    · timing: {timing}")
    if args.rendered:
        lines += ["", f"    Page rendered {args.rendered} from the "
                  f"[`nightly-bench`]({_NB}) snapshot."]
    lines += ["", "## Hardware ceilings", ""]
    if mach.known:
        lines += [
            "Utilisation is reported against the **attainable** ceiling — the "
            "spec-sheet peak scaled by what "
            f"[`benchmarks/hardware/`]({_GH}/tree/main/benchmarks/hardware) "
            "measures on this GPU. Reaching 100% means saturating the machine "
            "as microbenchmarks find it, not as the datasheet advertises it.",
            "",
            "| Resource | Spec-sheet peak | Attainable | Ratio |",
            "| --- | -: | -: | -: |",
            f"| HBM bandwidth | {bw_theo / 1e12:.2f} TB/s "
            f"| {bw_att / 1e12:.2f} TB/s | {bw_att / bw_theo:.0%} |",
        ]
        for key in ("fp8", "fp16", "bf16", "tf32"):
            theo, att = mach.tf_peak_by_key(key)
            if theo:
                lines.append(f"| Tensor core {key} | {theo / 1e12:.0f} TFLOP/s "
                             f"| {att / 1e12:.0f} TFLOP/s | {att / theo:.0%} |")
        lines += ["", "Source: "
                  f"[`src/tileops/perf/profiles/`]({_GH}/tree/main/src/tileops/perf/profiles)"]
    else:
        lines += [
            f"!!! warning \"No GPU profile for {args.gpu}\"", "",
            "    Utilisation, SOL and limit columns are blank: this renderer "
            "does not guess peaks. Add a profile under "
            f"[`src/tileops/perf/profiles/`]({_GH}/tree/main/src/tileops/perf/profiles) "
            "to fill them.",
        ]
    return lines


def legend_block() -> list[str]:
    return [
        "", "## How to read a row", "",
        "| Column | Meaning |",
        "| --- | --- |",
        "| **TFLOP/s** | FLOPs the workload requires ÷ measured latency. "
        "The FLOP count is the manifest `roofline` formula, not a "
        "hardware counter. |",
        "| **HBM TB/s** | Bytes the workload must move ÷ measured latency, "
        "from the same formula. |",
        "| **of peak** | The two utilisations: achieved ÷ attainable ceiling "
        "for that dtype / for HBM. |",
        "| **SOL** | Speed of light: the larger of the two utilisations — how "
        "close the kernel is to the resource that limits it. This is the one "
        "number to read if you read one. |",
        "| **Bound** | Which resource that is, at this workload's arithmetic "
        "intensity. `ᶜ` marks a workload whose "
        "achieved bandwidth exceeds the HBM peak, i.e. it ran out of cache, "
        "so the HBM ceiling understates its headroom. |",
        "| **Best alternative** | The fastest other implementation measured on "
        "the same workload. Reported with its own TFLOP/s and TB/s so it is "
        "comparable line by line, never as a bare ratio. Tier: unlabelled = "
        f"tuned library kernel, _{TIER_TORCH}_ = PyTorch native op, "
        f"_{TIER_REF}_ = eager reference composition (a weak bar — beating it "
        "by 10× is expected, not news). |",
        "| **Speedup / worst** | Median and worst-case alternative latency ÷ "
        "TileOPs latency, across the op's workloads. >1 means TileOPs is "
        "faster. A high median with a low worst means one shape regressed. |",
        "| **Test** | Correctness: ✅ passed · ❌ failed · ⏭️ all skipped · "
        f"`{EMPTY}` no test matched this op. |",
        "| **spread** | (p90 − p10) ÷ median latency over the timed samples — "
        f"noise around that measurement. `⚠` above {NOISY_SPREAD:.0f}%, where "
        "the median stops being a reliable summary. |",
        "| **AI** | Arithmetic intensity, FLOPs ÷ bytes. |",
        "",
        "**Status dot** on the op name is judged against the best "
        f"non-reference alternative where one was measured ({GREEN} ≥0.95× · "
        f"{YELLOW} 0.80–0.95× · {RED} <0.80×), otherwise against SOL "
        f"({GREEN} ≥70% · {YELLOW} 40–70% · {RED} <40%), otherwise "
        f"`{NA}`.",
        "",
        f"A `{EMPTY}` cell means an input to that metric was not recorded — "
        "see [Data gaps](#data-gaps). Aggregate rows are medians over the "
        "op's workloads; per-workload numbers are in each family's detail "
        "table.",
        "",
    ]


def rollup_block(rows: list[tuple], spreads: list[float]) -> list[str]:
    """How the 196 ops distribute over status and over judging basis."""
    by = Counter((s["status"], s["basis"]) for _, _, s, _, _ in rows)
    lines = ["", "## Where the library stands", "",
             "| | Judged against an alternative | Judged against the ceiling "
             "| Total |", "| --- | -: | -: | -: |"]
    labels = [(GREEN, f"{GREEN} at or ahead / ≥70% of SOL"),
              (YELLOW, f"{YELLOW} 0.80–0.95× / 40–70% of SOL"),
              (RED, f"{RED} below 0.80× / <40% of SOL"),
              (NA, f"{NA} no alternative and no ceiling resolved")]
    for status, label in labels:
        b, c = by[(status, "baseline")], by[(status, "sol")]
        n = b + c + by[(status, "none")]
        if n:
            lines.append(f"| {label} | {b or EMPTY} | {c or EMPTY} | {n} |")
    n_base = sum(v for (_, basis), v in by.items() if basis == "baseline")
    lines += ["", f"{n_base} of {len(rows)} ops have a non-reference "
              "alternative measured on the identical workload; the rest are "
              "judged against the hardware ceiling only, which is a weaker "
              "claim.", ""]
    if spreads:
        noisy = sum(1 for x in spreads if x > NOISY_SPREAD)
        lines += [f"Measurement noise: median p10→p90 spread "
                  f"{statistics.median(spreads):.1f}% of latency; "
                  f"{noisy} of {len(spreads)} workloads exceed "
                  f"{NOISY_SPREAD:.0f}% and carry `⚠`.", ""]
    return lines


def _rank_table(title: str, subset: list[tuple], note: str) -> list[str]:
    out = [f"### {title}", "", note, "",
           "| Op | Alternative | Speedup | worst | TileOPs TFLOP/s "
           "| its TFLOP/s | TileOPs TB/s | its TB/s | SOL |",
           "| --- | --- | -: | -: | -: | -: | -: | -: | -: |"]
    for op, module, s, _, ref in subset:
        out.append(
            f"| {s['status']} {_op_cell(op, module, ref)} "
            f"| {_rival_cell(s['rival'], s['rival_tier'])} "
            f"| {_speed(s['speedup'])} | {_speed(s['worst_speedup'])} "
            f"| {_sig(s['tflops'])} | {_sig(s['rival_tflops'])} "
            f"| {_sig(s['bw_tbs'])} | {_sig(s['rival_bw_tbs'])} "
            f"| {_pct(s['sol'])} |")
    out.append("")
    return out


def leaderboard_block(rows: list[tuple], k: int = 15) -> list[str]:
    """The two lists worth acting on: where an alternative is faster, and
    where TileOPs beats a tuned library kernel."""
    judged = [r for r in rows if r[2]["basis"] == "baseline"]
    if not judged:
        return []
    behind = sorted((r for r in judged if r[2]["speedup"] < 0.95),
                    key=lambda r: r[2]["speedup"])
    ahead_lib = sorted((r for r in judged if r[2]["rival_tier"] == TIER_LIB
                        and r[2]["speedup"] >= 1.0),
                       key=lambda r: -r[2]["speedup"])
    lines = ["", "## Two lists worth reading", ""]
    if behind:
        lines += _rank_table(
            f"Behind an alternative ({len(behind)} ops, worst {min(k, len(behind))})",
            behind[:k],
            "Sorted by median speedup. A `worst` far below the median means "
            "one shape carries the loss.")
    if ahead_lib:
        lines += _rank_table(
            f"Ahead of a tuned library kernel ({len(ahead_lib)} ops, "
            f"top {min(k, len(ahead_lib))})",
            ahead_lib[:k],
            "Restricted to library-kernel alternatives; a win over eager "
            "PyTorch is not listed here.")
    return lines


def gaps_block(rows: list[tuple], metrics_by_op: dict[str, list[dict]],
               unclassified: list[str], failures: list[dict],
               skips: list[dict]) -> list[str]:
    """Every reason a cell on this page is empty."""
    # An op counts as missing a quantity when any of its workloads lacks it:
    # the median hides a gap that the per-workload table shows.
    no_flops = sorted(op for op, ms in metrics_by_op.items()
                      if any(m["tflops"] is None for m in ms))
    no_bytes = sorted(op for op, ms in metrics_by_op.items()
                      if any(m["bw_tbs"] is None for m in ms))
    no_rival = sorted(op for op, _, s, _, _ in rows if s["rival"] is None)
    ref_only = sorted(op for op, _, s, _, _ in rows if s["rival_ref_only"])
    no_dtype = sorted(op for op, ms in metrics_by_op.items()
                      if any(m["compute_util"] is None and m["tflops"] is not None
                             for m in ms))
    no_test = sorted(op for op, _, _, tmark, _ in rows if tmark == EMPTY)
    n_wl = {"flops": sum(1 for ms in metrics_by_op.values() for m in ms
                         if m["tflops"] is None),
            "bytes": sum(1 for ms in metrics_by_op.values() for m in ms
                         if m["bw_tbs"] is None),
            "dtype": sum(1 for ms in metrics_by_op.values() for m in ms
                         if m["compute_util"] is None and m["tflops"] is not None)}

    lines = ["", "## Data gaps", "",
             "What is missing from this run, and what fills it.", "",
             "| Gap | Ops | Workloads | Effect on this page | Fix |",
             "| --- | -: | -: | --- | --- |"]

    def row(gap, ops, effect, fix, workloads=None):
        if not ops:
            return
        lines.append(f"| {gap} | {len(ops)} | {workloads if workloads else EMPTY} "
                     f"| {effect} | {fix} |")

    row("No FLOP count reported", no_flops,
        "TFLOP/s, AI and compute utilisation blank",
        "op `eval_roofline()` returns 0 FLOPs for the workload", n_wl["flops"])
    row("No byte count reported", no_bytes,
        "HBM TB/s, AI and bandwidth utilisation blank",
        "op `eval_roofline()` returns 0 bytes for the workload", n_wl["bytes"])
    row("Compute ceiling unresolved", no_dtype,
        "of-peak and SOL blank on the compute side",
        "record the executed dtype per workload instead of parsing it out of "
        "the workload name", n_wl["dtype"])
    row("No alternative measured", no_rival,
        "alternative columns blank; status falls back to SOL",
        "add a baseline to the op's benchmark")
    row("Only an eager reference measured", ref_only,
        "speedup shown but not used for status",
        "add a library or native-op baseline")
    row("No correctness test matched", no_test,
        "Test column blank — the numbers are unverified",
        "record the same `op` name in the op's test as in its benchmark")
    row("Unclassified baseline tag", unclassified,
        "counted as a tuned library kernel, which may overstate the bar",
        "add the tag to the tier table in `scripts/gen_bench_pages.py`")
    if failures:
        lines.append(f"| Benchmark errored | {EMPTY} | {len(failures)} | workload "
                     "absent from every table | fix the benchmark |")
    if skips:
        lines.append(f"| Benchmark skipped | {EMPTY} | {len(skips)} | workload "
                     "absent from every table | supply the missing dependency "
                     "or kernel |")
    lines.append("")

    def detail(title, ops):
        if not ops:
            return
        lines.extend([f'??? note "{title} ({len(ops)})"', "",
                      "    " + ", ".join(f"`{_md_code(o)}`" for o in ops), ""])

    detail("Ops reporting no FLOP count", no_flops)
    detail("Ops reporting no byte count", no_bytes)
    detail("Ops with no alternative implementation measured", no_rival)
    detail("Ops compared only against an eager reference", ref_only)
    detail("Ops with no correctness test in this run", no_test)
    if unclassified:
        detail("Unclassified baseline tags", unclassified)

    for title, items in (("Benchmarks that errored", failures),
                         ("Benchmarks skipped", skips)):
        if not items:
            continue
        lines.extend([f'??? note "{title} ({len(items)})"', "",
                      "    | Workload | Reason |", "    | --- | --- |"])
        for it in items:
            msg = (it.get("message") or "").strip()
            lines.append(f"    | `{_md_code(it['name'])}` | "
                         f"{_md(msg[:160]) or EMPTY} |")
        lines.append("")
    return lines


ROADMAP = [
    ("Achieved DRAM traffic", "measured bytes moved, vs the formula's",
     "would separate a kernel that moves more bytes than it must from one "
     "that is genuinely bandwidth-bound",
     "CUPTI metric collection in the benchmark run"),
    ("L2 hit rate", "share of traffic served by cache",
     "would replace the `ᶜ` marker with a number and give cache-resident "
     "workloads a real ceiling",
     "CUPTI metric collection"),
    ("Achieved occupancy", "resident warps ÷ maximum",
     "would explain a low SOL that neither utilisation accounts for",
     "CUPTI metric collection"),
    ("Tensor core utilisation", "pipe-active cycles ÷ elapsed",
     "would show whether a compute-bound kernel is limited by the MMA pipe "
     "or by everything around it",
     "CUPTI metric collection"),
    ("Energy per workload", "joules per call",
     "would rank ops by efficiency rather than latency alone",
     "NVML power sampling around the timed region"),
    ("Numerical error against fp64", "worst relative error per workload",
     "would put accuracy next to speed, so a fast kernel that lost precision "
     "is visible here",
     "per-workload error recorded in the benchmark, not only in tests"),
]


def roadmap_block() -> list[str]:
    lines = ["", "## Metrics this page does not have yet", "",
             "Designed columns whose inputs no part of the pipeline records. "
             "They are named here rather than shown as empty columns on every "
             "row.", "",
             "| Metric | Definition | What it would answer | Blocked on |",
             "| --- | --- | --- | --- |"]
    for name, defn, answers, blocker in ROADMAP:
        lines.append(f"| {name} | {defn} | {answers} | {blocker} |")
    lines.append("")
    return lines


# --- Main ------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-xml", required=True)
    ap.add_argument("--test-xml")
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--date", default="unknown")
    ap.add_argument("--gpu", default="unknown")
    ap.add_argument("--rendered", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ref = args.commit if args.commit and args.commit != "unknown" else "main"

    configs, failures, skips = parse_bench_xml(args.bench_xml)
    tests = (parse_test_xml(args.test_xml)
             if args.test_xml and os.path.exists(args.test_xml) else {})
    mach = Machine(load_gpu_profile(args.gpu))

    unclassified = sorted({
        t for c in configs for t in c["impls"]
        if not t.startswith("tileops") and t not in _KNOWN_TAGS})
    timing = Counter(
        c["impls"].get("tileops", {}).get("timing") for c in configs).most_common(1)
    timing = timing[0][0] if timing else None

    by_op: dict[str, list[dict]] = defaultdict(list)
    module_of, cfgs_of = {}, defaultdict(list)
    for c in configs:
        by_op[c["op"]].append(config_metrics(c, mach))
        cfgs_of[c["op"]].append(c)
        module_of.setdefault(c["op"], c["op_module"])

    rows_by_fam: dict[str, list[tuple]] = defaultdict(list)
    all_rows = []
    for op, ms in by_op.items():
        s = op_summary(ms)
        row = (op, module_of.get(op), s, _test_mark(tests.get(op)), ref)
        rows_by_fam[family_of(op, module_of.get(op))].append(row)
        all_rows.append(row)

    ordered = ([f for f in FAMILY_ORDER if f in rows_by_fam]
               + sorted(f for f in rows_by_fam if f not in FAMILY_ORDER))
    spreads = [m["spread_pct"] for ms in by_op.values() for m in ms
               if m["spread_pct"] is not None]
    lines = header_block(args, mach, len(by_op), len(ordered), len(configs), timing)
    lines += legend_block()
    lines += rollup_block(all_rows, spreads)
    lines += leaderboard_block(all_rows)

    rank = {GREEN: 0, YELLOW: 1, RED: 2, NA: 3}
    for fam in ordered:
        rows = sorted(rows_by_fam[fam],
                      key=lambda r: (rank.get(r[2]["status"], 9), r[0]))
        lines += [f"## {FAMILY_TITLE.get(fam, fam)} "
                  f"<small>({len(rows)} ops)</small>", "", *SUMMARY_HEADER]
        for op, module, s, tmark, _ in rows:
            lines.append(summary_row(op, module, s, tmark, ref))
        lines += ["", '??? note "Per-workload detail"', "", *DETAIL_HEADER]
        for op, *_ in rows:
            for cfg, m in sorted(zip(cfgs_of[op], by_op[op]),
                                 key=lambda z: z[0]["config"]):
                lines.append(detail_row(op, cfg, m))
        lines.append("")

    lines += gaps_block(all_rows, by_op, unclassified, failures, skips)
    lines += roadmap_block()

    out = args.out or os.path.join(REPO, "docs", "benchmarks", "index.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out}: {len(by_op)} ops, {len(ordered)} families, "
          f"{len(configs)} workloads, {len(failures)} failed, {len(skips)} skipped")


if __name__ == "__main__":
    main()
