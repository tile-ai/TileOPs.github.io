#!/usr/bin/env python3
"""Render the Benchmarks section from a nightly benchmark XML snapshot.

Output is one overview page, one page explaining the numbers, and five data
pages grouped by op domain. `hooks.py` puts them into the site nav in that
order.

These pages answer one question per workload: **how does TileOPs compare to the
fastest other implementation of the same op on that workload?**

Rules this renderer follows:

  * Every number belongs to one workload. Nothing is aggregated across an op's
    workloads: they span shapes and dtypes orders of magnitude apart, so a
    median matches no reproducible run and a mean ratio hides which shape is
    behind.
  * `Ratio` is the column right after the workload name, and its colour is the
    verdict — red behind, plain ink level, green ahead, grey against an eager
    reference only.
  * The compared quantity is ``device_busy_ms``: the time the device spent
    executing the call's kernels. A single-kernel call has no gap between
    kernels by construction, so comparing spans would charge a multi-kernel
    implementation for the host's launch latency and credit a fused one for
    nothing it did.
  * Two questions, two columns: ``Ratio`` says whether someone else's kernel
    does the same work faster; ``SOL`` says how much faster the hardware allows
    anyone to go. The SOL arithmetic is imported from the TileOPs checkout's
    roofline tool (M5), never re-derived here.
  * Every baseline present in the data is shown. Baselines are tiered
    (library kernel / PyTorch native op / eager reference), never discarded:
    a tag the tier table does not know is reported as unclassified. Only the
    first two tiers can rate an op — beating an eager composition of PyTorch
    ops is not a result, so an op with no better rival stays unrated.
  * A metric whose input is missing renders as the empty marker. No metric is
    silently substituted.

Usage:
    python scripts/gen_bench_pages.py --bench-xml <xml> [--test-xml <xml>] \
        [--meta <meta.json>] --commit <sha> --date <YYYY-MM-DD> --gpu "NVIDIA H200"
"""
from __future__ import annotations

import argparse
import html
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

# How an op stands against the fastest real alternative measured on its
# workloads. The verdict is carried by the colour of the ratio itself rather
# than by a separate status glyph, so a reader gets it from the number they
# were already reading.
AHEAD, PAR, BEHIND, UNRATED = "ahead", "par", "behind", "unrated"
PAR_BAND = (0.95, 1.05)  # inside this the two implementations are level
NA = "—"
EMPTY = "·"  # a metric whose input was not recorded

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


# --- Workload dtype --------------------------------------------------------
_DTYPE_TOKENS = ("fp8", "bfloat16", "bf16", "float16", "fp16", "float32")


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def dtype_of(config_name: str) -> str | None:
    """The dtype token a workload name carries, if any."""
    n = config_name.lower()
    for tok in _DTYPE_TOKENS:
        if tok in n:
            return tok
    return None


# --- XML parsing -----------------------------------------------------------
# Property names are <tag>_<metric>. Parsing is generic over the metric suffix
# so a metric added on the TileOPs side appears here without a code change.
# Longer suffixes come first: device_busy_p10_ms must not match as latency_ms.
_METRIC_SUFFIXES = (
    "device_busy_p10_ms", "device_busy_p90_ms", "device_busy_ms",
    "latency_p10_ms", "latency_p90_ms", "latency_ms", "gap_ms",
    "uncounted_copy_ms", "bandwidth_tbs", "tflops", "ratio", "n_kernels",
    "n_samples", "flops", "bytes", "compute_roof", "dtype", "timing",
    "variant",
)
_NUMERIC_METRICS = {
    "device_busy_ms", "device_busy_p10_ms", "device_busy_p90_ms",
    "latency_ms", "latency_p10_ms", "latency_p90_ms", "gap_ms",
    "uncounted_copy_ms", "tflops", "bandwidth_tbs", "ratio", "flops",
    "bytes", "n_kernels", "n_samples",
}
# The alias the benchmark writes for its first baseline. Dropped whole: it
# duplicates an implementation under a name none has, and its key set grows.
_LEGACY_TAG = "baseline"


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
            if key in ("op", "op_module"):
                continue
            for suf in _METRIC_SUFFIXES:
                if key.endswith("_" + suf):
                    tag = key[: -len(suf) - 1]
                    if tag == _LEGACY_TAG:
                        break
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


# --- Speed of light ----------------------------------------------------------
# The SOL reading is computed by TileOPs' own roofline tool (M5), imported from
# the ./TileOPs checkout: the arithmetic, its exclusion rules and its verdict
# thresholds must have exactly one implementation. Without the checkout, or
# without a GPU profile for the measured device, every SOL cell renders as the
# empty marker.


def load_sol_engine(gpu: str):
    """(nightly_report module, GPU profile) or (None, None) with a warning."""
    try:
        sys.path.insert(0, os.path.join(TILEOPS, "src"))
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "tileops_nightly_report",
            os.path.join(TILEOPS, "scripts", "nightly_report.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from tileops.perf.profile import find_profile
        profile = find_profile(gpu)
    except Exception as exc:  # noqa: BLE001 — degrade to an empty column
        print(f"warning: SOL column disabled ({exc})", file=sys.stderr)
        return None, None
    if profile is None:
        print(f"warning: SOL column disabled (no GPU profile matches {gpu!r})",
              file=sys.stderr)
        return None, None
    return mod, profile


def sol_of(tl: dict, engine) -> dict | None:
    """M5's reading for one tileops row, with the verdict already graded.

    The verdict grades against M5's own thresholds so the page and the nightly
    report can never disagree on where the at-ceiling line sits.
    """
    mod, profile = engine
    if mod is None:
        return None
    sol = mod._compute_sol({"tileops_" + k: v for k, v in tl.items()}, profile)
    if sol is None:
        return None
    if sol["impossible"] or sol["efficiency"] > mod.SOL_ANOMALY:
        verdict = "anomaly"
    elif sol["latency_bound"]:
        verdict = "lat"
    else:
        green = (mod.SOL_GREEN_MEMORY if sol["bound"] == "memory"
                 else mod.SOL_GREEN_COMPUTE)
        verdict = "ceiling" if sol["efficiency"] >= green else "below"
    return {**sol, "verdict": verdict}


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


def workload_metrics(w: dict, sol_engine=(None, None)) -> dict:
    """Derive every displayed metric for one benchmarked workload."""
    tl = w["impls"].get("tileops", {})
    busy = _busy_of(tl)
    tflops = _pos(tl.get("tflops"))

    m = {
        "busy_ms": busy,
        "tflops": tflops,
        "dtype": tl.get("dtype") or dtype_of(w["config"]),
        "n_samples": tl.get("n_samples"),
        "variant": tl.get("variant"),
        "sol": sol_of(tl, sol_engine),
    }

    rivals = {}
    for tag, d in w["impls"].items():
        if tag.startswith("tileops"):
            continue
        b_busy = _busy_of(d)
        if not b_busy:
            continue
        # The benchmark computes the ratio before rounding its times for the
        # XML, so prefer it: a sub-microsecond kernel loses several percent to
        # the write precision of the times alone.
        computed = (b_busy / busy) if busy else None
        recorded = _pos(d.get("ratio"))
        rivals[tag] = {
            "tier": tier_of(tag), "busy_ms": b_busy,
            "speedup": recorded or computed,
            "computed_ratio": computed, "recorded_ratio": recorded,
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
    """What can be said about an op as a whole: its workload count and whether
    any workload had a real alternative. No aggregate is rendered — the
    geometric-mean ratio here only orders the op sections.
    """
    s = {"workloads": len(metrics)}

    tag, ratio = best_rival(metrics, (TIER_LIB, TIER_TORCH))
    ref_only = False
    if tag is None:
        tag, ratio = best_rival(metrics, (TIER_REF,))
        ref_only = tag is not None
    s.update(rival=tag, speedup=ratio, rival_ref_only=ref_only)

    # Only a real alternative measured on the identical workload says anything
    # about the gap to the state of the art. An eager reference does not: beating
    # a naive composition of PyTorch ops is not a result, so those ops stay
    # unrated rather than being scored against a bar nobody competes at.
    if ratio is not None and not ref_only:
        lo, hi = PAR_BAND
        s["status"] = AHEAD if ratio >= hi else PAR if ratio >= lo else BEHIND
    else:
        s["status"] = UNRATED
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


def _sig_ms(x) -> str:
    """A time in ms. Fixed decimals would round a small kernel away, so
    anything under 10 us keeps significant digits instead."""
    if x is None:
        return EMPTY
    return f"{x:.4f}" if x >= 0.01 else f"{x:.3g}"


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


def _ratio_cell(ratio: float | None, rated: bool = True) -> str:
    """The gap to the alternative, coloured by which side of parity it lands on.

    Red for behind, plain ink for level, green for ahead — the reader gets the
    verdict from the number itself instead of a legend.

    `rated=False` for a ratio against an eager reference only: the number is
    still shown, because it says the kernel does something, but it stays grey.
    Painting a 18x win over a naive composition of PyTorch ops the same green as
    a win over a tuned library kernel would overstate it.
    """
    if ratio is None:
        return f'<span class="perf-none">{NA}</span>'
    if not rated:
        return f'<span class="perf-unrated">{_speed(ratio)}</span>'
    lo, hi = PAR_BAND
    cls = "perf-ahead" if ratio >= hi else "perf-par" if ratio >= lo else "perf-behind"
    return f'<span class="{cls}">{_speed(ratio)}</span>'


def _sol_cell(sol: dict | None) -> str:
    """Share of the achievable ceiling, with the side that binds.

    Green means physics allows little more and the workload is done; plain ink
    is headroom. A latency-bound row and a reading above the ceiling both grey
    out — the first is a workload the model cannot judge, the second a formula
    or calibration error, and neither must read as a fast kernel.
    """
    if sol is None:
        return EMPTY
    pct = f"{sol['efficiency']:.0%}"
    bound = "M" if sol["bound"] == "memory" else "C"
    if sol["verdict"] == "anomaly":
        return f'<span class="perf-unrated">⚠ {pct}</span>'
    if sol["verdict"] == "lat":
        return '<span class="perf-unrated">lat-bound</span>'
    if sol["verdict"] == "ceiling":
        return f'<span class="perf-ahead">{pct} {bound}</span>'
    return f"{pct} {bound}"


# --- Data tables -----------------------------------------------------------

# Every number belongs to one workload; there is no per-op aggregate, since a
# median over shapes orders of magnitude apart matches no reproducible run.
# HTML rather than Markdown because Markdown cannot span a heading across
# columns. No class on the `<table>`: the Material and `extra.css` rules key off
# `table:not([class])`.
DETAIL_HEADER = (
    "<table>",
    "<thead>",
    "<tr>",
    # `colsep` draws the hairline between the workload name and the numbers.
    # Its own cell spans both header rows, so the rule has no gap.
    '<th rowspan="2" class="colsep">Workload</th>',
    "<th>Ratio</th>",
    "<th>Device time</th>",
    '<th colspan="2">Alternatives</th>',
    "<th>Throughput</th>",
    "<th>SOL</th>",
    "</tr>",
    # The second row carries what a header word cannot: the unit, and which way
    # the ratio divides. Every numeric column states its own on the same line.
    "<tr>",
    '<th class="subhead">alt / ours</th>',
    '<th class="subhead">ms</th>',
    '<th class="subhead">name</th>',
    '<th class="subhead">ms</th>',
    '<th class="subhead">TFLOP/s</th>',
    '<th class="subhead">of ceiling</th>',
    "</tr>",
    "</thead>",
    "<tbody>",
)
DETAIL_FOOTER = ("</tbody>", "</table>")


def _stack(cells: list[str]) -> str:
    """One line per alternative, fastest first. Both sub-columns stack in the
    same order, so they read across without a nested table.

    Only the first line stays at full strength: it is the one `Ratio` divides
    by. The slower alternatives are context, and reading them as equals costs a
    reader the moment of finding which bar was actually cleared.
    """
    if not cells:
        return EMPTY
    head, *rest = cells
    if not rest:
        return head
    tail = "<br>".join(f'<span class="alt-slow">{c}</span>' for c in rest)
    return f"{head}<br>{tail}"


def detail_row(w: dict, m: dict) -> str:
    ordered = sorted(m["rivals"].items(), key=lambda kv: kv[1]["busy_ms"])
    names = _stack([f"<code>{html.escape(t)}</code>" for t, _ in ordered])
    times = _stack([_sig_ms(r["busy_ms"]) for _, r in ordered])
    # Against the fastest non-reference alternative, so a win over an eager
    # reference is not painted as a win over a real one — see `_ratio_cell`.
    real = [r for _, r in ordered if r["tier"] != TIER_REF and r["speedup"]]
    weak = [r for _, r in ordered if r["tier"] == TIER_REF and r["speedup"]]
    gap = _ratio_cell(real[0]["speedup"] if real else
                      weak[0]["speedup"] if weak else None,
                      rated=bool(real))
    return (
        "<tr>"
        f'<td class="colsep"><code>{html.escape(w["config"])}</code></td>'
        f"<td>{gap}</td>"
        f"<td>{_sig_ms(m['busy_ms'])}</td>"
        f"<td>{names}</td>"
        f"<td>{times}</td>"
        f"<td>{_sig(m['tflops'])}</td>"
        f"<td>{_sol_cell(m['sol'])}</td>"
        "</tr>"
    )


# --- Pages -----------------------------------------------------------------

# Environment keys in the order the overview table shows them; anything else
# recorded in meta.json is appended so a newly published fact is never dropped.
ENV_ORDER = ["image", "gpu", "driver", "cuda", "torch", "tilelang", "timer"]
# The warmup and measurement budgets are deliberately not published as facts
# about a number. They are a per-implementation time budget the harness fills
# with as many samples as fit, so "25 ms" reads as if a call took 25 ms.
_ENV_HIDE = {"warmup_ms", "repeat_ms"}


def env_block(meta: dict, timing: str | None) -> list[str]:
    """The stack the numbers were produced on, from the published meta.json."""
    # A nested value is an inventory, not a fact for this table.
    env = {k: v for k, v in (meta.get("environment") or {}).items()
           if not isinstance(v, (dict, list)) and k not in _ENV_HIDE}
    packages = meta.get("packages") or (meta.get("environment") or {}).get("packages") or {}
    if timing and "timer" not in env:
        env["timer"] = timing
    lines = ["## Environment", ""]
    if not env:
        lines += [
            '!!! warning "The run did not publish its environment"', "",
            "    Without it a number on these pages cannot be tied to the "
            "machine and the stack that produced it. The nightly's publish "
            f"step fills this in through [`meta.json`]({_NB}).", "",
        ]
    else:
        keys = ([k for k in ENV_ORDER if k in env]
                + [k for k in env if k not in ENV_ORDER])
        lines += ["| | |", "| --- | --- |"]
        lines += [f"| {_md(k)} | `{_md_code(env[k])}` |" for k in keys]
        # A version the inventory carries is published, wherever it sits.
        missing = [k for k in ("image", "driver", "cuda", "torch", "tilelang")
                   if k not in env and k not in packages]
        if missing:
            lines += ["", "Not published by this run: "
                      + ", ".join(f"`{k}`" for k in missing) + "."]
    # The full installed-package inventory is not published here. It is a few
    # hundred rows nobody reads to understand a benchmark number, and the
    # versions that do matter are named in the table above. The snapshot itself
    # carries the inventory for anyone reproducing a run.
    return lines + [""]


def method_block() -> list[str]:
    """How the numbers were taken. Fixed policy of the benchmark layer.

    Kept to what changes how a number should be read. The reasoning behind the
    compared quantity lives on the reading page, not here.
    """
    return [
        "## Method", "",
        "- **One process, common inputs.** Every implementation of an op is "
        "timed on the same tensors in the same process, in forward and then "
        "reversed order so drift does not land on whichever ran last.",
        "- **A fixed warmup and measurement budget** per implementation, "
        "reported as the median over however many samples fit in it, with L2 "
        "cleared between iterations.",
        "- **Compilation and workspace setup excluded.**",
        "- **Device time is what is compared** — the union of the intervals the "
        "device spent executing the call's kernels, collected through CUPTI. A "
        "run that cannot collect device activity fails rather than falling back "
        "to a different clock.",
        "",
    ]


def index_page(args, meta: dict, rows: list[tuple],
               by_page: dict, timing: str | None,
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

    lines = head + env_block(meta, timing) + method_block()

    # What the run covers and how far to trust it — the qualifications a reader
    # needs before reading any single number off a data page.
    by = Counter(s["status"] for _, _, s, _, _ in rows)
    rated = len(rows) - by[UNRATED]
    lines += ["## Coverage", "",
              f"- **{rated} of {len(rows)} ops** are measured against a real "
              "alternative — a tuned library kernel or a native PyTorch op — on "
              "the identical workload. The rest run against an eager reference "
              "only, which is not a bar worth reporting a win against."]
    if n_failed or n_skipped:
        lines.append(f"- **Absent from every table**: {n_failed} workloads "
                     f"errored and {n_skipped} were skipped in this run.")
    lines += ["", "[How these numbers are taken](reading.md)", ""]

    # Entry table into the data pages. Coverage only: the per-page verdict
    # tallies belong on the page that shows the rows behind them.
    lines += ["## Data", "",
              "| Page | Ops | Workloads |", "| --- | --- | --- |"]
    for slug, title, _ in DATA_PAGES:
        page_rows = by_page.get(slug, [])
        if not page_rows:
            continue
        n_w = sum(s["workloads"] for _, _, s, _, _ in page_rows)
        lines.append(f"| [{title}]({slug}.md) | {len(page_rows)} | {n_w} |")
    return "\n".join(lines) + "\n"


def reading_page() -> str:
    lo, hi = PAR_BAND
    lines = [
        "# How these numbers are taken", "",
        "Every data page answers one question: **how does TileOPs compare to the "
        "fastest alternative implementation of the same op, on the same "
        "workload?** Each op gets one table, with one row per workload. Nothing "
        "is averaged across workloads: every number on the page belongs to a "
        "single shape and dtype.", "",
        "## The colour is the verdict", "",
        "| | Meaning |",
        "| --- | --- |",
        f'| <span class="perf-behind">0.74×</span> | Slower than the '
        f"alternative — below {lo:.2f}×. |",
        f'| <span class="perf-par">1.02×</span> | Level with it — '
        f"{lo:.2f}–{hi:.2f}×, inside measurement noise. |",
        f'| <span class="perf-ahead">1.42×</span> | Faster than it — '
        f"{hi:.2f}× and above. |",
        f'| <span class="perf-unrated">18.06×</span> | No fast alternative to '
        f"compare against — only a functional reference (a name ending in "
        f"`-{TIER_REF}`). The number means little. |",
        f'| <span class="perf-none">{NA}</span> | No alternative at all ran on '
        "this workload. |",
        "",
        "A ratio is the alternative's device time divided by ours, so **above 1 "
        "means TileOPs is faster**.", "",
        "## Columns", "",
        "| Column | Meaning |",
        "| --- | --- |",
        "| **Workload** | The shape and dtype the row was measured on, as the "
        "benchmark names it. |",
        "| **Ratio** | `alt / ours` — the fastest alternative's device time "
        "divided by ours, the one number the colour grades. |",
        "| **Device time** | Milliseconds the device spent executing the call's "
        "kernels — the union of their intervals. Every comparison on these "
        "pages uses it. |",
        "| **Alternatives** | One line per other implementation measured on this "
        "workload, fastest first, with its own device time in ms. A tuned "
        f"library kernel (`fla`, `mamba`, `fa3`, `triton`, …), a native PyTorch "
        f"op (`{TIER_TORCH}`), or a name ending in `-{TIER_REF}` — an eager "
        "composition of PyTorch ops, which is not a bar worth reporting a win "
        "against. Divide any of them by our device time to get the ratio "
        "against that one. |",
        "| **Throughput** | TFLOP/s: required FLOPs / device time. The count is "
        "analytic — the op's `eval_roofline` formula evaluated on the workload's "
        "own shapes, not a hardware counter — so it counts the work the problem "
        "demands, not the instructions the kernel issued. Padding, recompute or "
        "a masked-out tile is therefore invisible here, and the figure is only "
        "comparable between implementations of the same op on the same "
        "workload. |",
        "| **SOL** | Share of the algorithmic speed-of-light: the fastest time "
        "physics allows for the workload, divided by our device time. The "
        "`Ratio` column says whether someone is faster today; SOL says how much "
        "faster anyone could ever be. Details below. |",
        "",
        "How that device time is measured — what it counts, what it leaves out, "
        "and where it refuses to produce a number — is in "
        "[Benchmark Timing](../timing.md).",
        "",
        "Each op's heading carries its workload count and its test outcome "
        "(✅ passed · ❌ failed · ⏭️ all skipped · "
        f"`{EMPTY}` no test matched).",
        "",
        "## Speed of light", "",
        "**SOL** is *algorithmic* speed-of-light efficiency: "
        "`max(bytes / bandwidth, FLOPs / compute roof) / device time`, priced "
        "against the machine's *calibrated* ceilings — the bandwidth and "
        "compute rates microbenchmarks actually reach on this GPU, not the "
        "spec sheet. 100% means no implementation of this algorithm on this "
        "hardware can be faster.", "",
        "Three statements delimit what a reading means:", "",
        "1. **Bytes are the algorithm's minimum traffic** — each input read "
        "once, each output written once — not the DRAM traffic the kernel "
        "generated. A kernel that moves data twice scores low; that is the "
        "point.",
        "2. **FLOPs follow the TileOPs counting convention** (a transcendental "
        "counts as one), not per-instruction hardware cost; the metric does "
        "not certify a special-function-bound kernel as at its limit.",
        "3. **The compute roof is the unit an optimal implementation would "
        "use** — declared per op, never inferred from the running kernel, so a "
        "kernel on the wrong unit is measured against the right ceiling.",
        "",
        "| | Meaning |",
        "| --- | --- |",
        '| <span class="perf-ahead">92% M</span> | At the achievable ceiling — '
        "memory-bound from 90%, compute-bound from 80% (the compute "
        "calibration is noisier). Optimizing further buys at most the "
        "remainder. |",
        "| 63% M | Headroom remains. `M`/`C` names the bound: memory traffic "
        "or compute throughput. |",
        '| <span class="perf-unrated">lat-bound</span> | The workload is too '
        "small for the model to judge — launch overhead dominates the "
        "measurement, not the roofline. |",
        '| <span class="perf-unrated">⚠ 108%</span> | Above the calibrated '
        "ceiling: the formula or the calibration is wrong. Never read it as a "
        "fast kernel. |",
        f"| `{EMPTY}` | An input is missing: no roofline formula, a non-CUPTI "
        "timing, or no GPU profile for the device. |",
        "",
        "The model, its thresholds and the formula-audit machinery are "
        "specified in TileOPs "
        f"[`docs/design/roofline.md`]({_GH}/blob/main/docs/design/roofline.md); "
        "the page imports that implementation rather than re-deriving it.",
        "",
        "## Empty cells", "",
        f"`{EMPTY}` means an input to that metric was not recorded, never that "
        "the value is zero: the op reported no FLOP count for that workload, or "
        "no alternative ran on it.",
        "",
    ]
    return "\n".join(lines) + "\n"


def data_page(title: str, fams: list[str], rows_by_fam: dict,
              metrics_by_op: dict, workloads_of: dict, ref: str) -> str:
    # Widest lead first, then level, then behind, then the unrated. Every op is
    # listed either way; this only decides what a reader meets first.
    rank = {AHEAD: 0, PAR: 1, BEHIND: 2, UNRATED: 3}
    present = [f for f in fams if rows_by_fam.get(f)]
    n_ops = sum(len(rows_by_fam[f]) for f in present)
    n_workloads = sum(s["workloads"] for f in present
                      for _, _, s, _, _ in rows_by_fam[f])
    tally = " · ".join(
        f"{FAMILY_TITLE.get(f, f)} {len(rows_by_fam[f])}" for f in present)
    lines = [f"# {title}", "",
             f"**{n_ops} ops, {n_workloads} workloads** — {tally}."
             if len(present) > 1 else
             f"**{n_ops} ops, {n_workloads} workloads.**", "",
             "One table per op, one row per workload. `Ratio` is the fastest "
             "other implementation's device time divided by ours, so "
             '<span class="perf-ahead">green</span> is faster than it, '
             '<span class="perf-par">plain</span> is level with it, '
             '<span class="perf-behind">red</span> is slower. Times are in ms. '
             "[How these numbers are taken](reading.md).", ""]
    for fam in fams:
        rows = rows_by_fam.get(fam)
        if not rows:
            continue
        # Within a band, the widest margin first.
        rows = sorted(rows, key=lambda r: (rank.get(r[2]["status"], 9),
                                           -(r[2]["speedup"] or 0), r[0]))
        lines += [f"## {FAMILY_TITLE.get(fam, fam)}", ""]
        # One table per op rather than one per family: the op name would
        # otherwise repeat down the widest column of every row. The `datatable`
        # wrapper is a styling hook — see extra.css.
        for op, module, s, tmark, _ in rows:
            # `·` after the separator reads as punctuation, so it is dropped.
            note = f"{s['workloads']} workloads"
            if tmark != EMPTY:
                note += f" · {tmark}"
            lines += [f"### {_op_cell(op, module, ref)} <small>({note})</small>",
                      "",
                      # No `markdown="1"`, and no blank line until `</div>`: a
                      # blank line would end the raw-HTML block mid-table.
                      '<div class="datatable">', *DETAIL_HEADER]
            for w, m in sorted(zip(workloads_of[op], metrics_by_op[op]),
                               key=lambda z: z[0]["config"]):
                lines.append(detail_row(w, m))
            lines += [*DETAIL_FOOTER, "</div>", ""]
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

    unclassified = sorted({
        t for w in workloads for t in w["impls"]
        if not t.startswith("tileops") and t not in _KNOWN_TAGS})
    timings = Counter(w["impls"].get("tileops", {}).get("timing")
                      for w in workloads).most_common(1)
    timing = timings[0][0] if timings else None

    sol_engine = load_sol_engine(args.gpu)

    metrics_by_op: dict[str, list[dict]] = defaultdict(list)
    workloads_of: dict[str, list[dict]] = defaultdict(list)
    module_of: dict[str, str | None] = {}
    for w in workloads:
        metrics_by_op[w["op"]].append(workload_metrics(w, sol_engine))
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
    out_dir = args.out_dir or os.path.join(REPO, "docs", "benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    pages = {
        "index.md": index_page(args, meta, all_rows, by_page,
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
