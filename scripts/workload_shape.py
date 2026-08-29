#!/usr/bin/env python3
"""Resolve a benchmarked workload to the shape and dtype of each input tensor.

The nightly benchmark snapshot names a workload only by its pytest parameter id
(``hidden-state-prefill-float16``). The shapes behind that name live in the
TileOPs spec manifest, ``src/tileops/manifest/*.yaml``, which the docs build
already checks out. This module joins the two: a manifest workload entry is
addressed by ``<label>-<dtype>``, which is exactly the id the benchmark writes.

What a workload entry carries is not uniform, so three sources are read in this
order and merged into one description:

  1. ``<tensor>_shape`` keys — a tensor's shape, given outright.
  2. ``signature.inputs.<tensor>.shape`` templates (``"[batch, seq_len, heads]"``)
     evaluated against the workload's own scalars, for the ops that declare
     dimensions rather than shapes.
  3. whatever scalar dimensions are left, named as the manifest names them.

Anything the manifest declares as a parameter rather than a dimension
(``is_causal``, ``page_size``) is reported separately and only when it differs
from the signature's default: a workload that takes the default is the ordinary
case and saying so on every row costs more than it tells.
"""
from __future__ import annotations

import ast
import glob
import os
import re
from collections import OrderedDict

import yaml

# The row already states one dtype, so a key that only repeats it is dropped.
_DTYPE_KEYS = {"dtype", "dtypes", "in_dtype", "out_dtype", "input_dtype",
               "output_dtype", "cache_dtype"}
_LABEL_KEYS = {"label"}

# Abbreviations, so a dtype costs a few characters in a cell rather than a word.
DTYPE_ABBR = {
    "float16": "f16", "bfloat16": "bf16", "float32": "f32", "float64": "f64",
    "float8_e4m3fn": "fp8e4m3", "float8_e5m2": "fp8e5m2",
    "int8": "i8", "int16": "i16", "int32": "i32", "int64": "i64",
    "uint8": "u8", "bool": "bool",
}


def abbr_dtype(name: str) -> str:
    return DTYPE_ABBR.get(name, name)


class Spec:
    """One benchmarked workload, described in the manifest's own terms."""

    def __init__(self, label, dtype, tensors, dims, params):
        self.label = label          # "hidden-state-prefill"
        self.dtype = dtype          # "float16" — the workload's dtype row
        self.tensors = tensors      # [(names, "2048×4096", dtype or None)]
        self.dims = dims            # [("heads", "32"), ...]
        self.params = params        # [("is_causal", "true"), ...]

    def __bool__(self) -> bool:
        return bool(self.tensors or self.dims or self.params)


# --- Manifest ---------------------------------------------------------------


def load_manifest(directory: str) -> dict:
    """Every op entry in a manifest directory, keyed by op name."""
    ops: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.yaml"))):
        with open(path, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        for op, entry in doc.items():
            if isinstance(entry, dict):
                ops[op] = entry
    return ops


# --- Shape templates --------------------------------------------------------
# A template is an expression over the workload's own scalars, so it is parsed
# rather than executed: only integer arithmetic over names the workload defines
# resolves, and anything else leaves the tensor undescribed instead of guessing.

_ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                  ast.Name, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.USub,
                  ast.FloorDiv, ast.Div, ast.Mod)


def _eval_dim(expr: str, scope: dict):
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return None
        if isinstance(node, ast.Name) and node.id not in scope:
            return None
        if isinstance(node, ast.Constant) and not isinstance(node.value, int):
            return None
    try:
        value = eval(compile(tree, "<manifest>", "eval"), {"__builtins__": {}}, scope)
    except Exception:
        return None
    return value if isinstance(value, int) else None


def _eval_template(template: str, scope: dict):
    """``"[batch, seq_len, heads]"`` -> ``[1, 8192, 32]``, or None."""
    body = template.strip()
    if not (body.startswith("[") and body.endswith("]")):
        return None
    dims = []
    for part in _split_dims(body[1:-1]):
        value = _eval_dim(part, scope)
        if value is None:
            return None
        dims.append(value)
    return dims or None


def _split_dims(body: str) -> list[str]:
    """Split on commas that are not inside brackets or parentheses."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(body):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(body[start:i])
            start = i + 1
    parts.append(body[start:])
    return [p for p in (p.strip() for p in parts) if p]


# --- Formatting -------------------------------------------------------------


def fmt_shape(dims) -> str:
    return "×".join(str(d) for d in dims)


def _fmt_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        # A per-request length list: [512, 512, 512, 512] is four of one length.
        if value and all(isinstance(x, int) for x in value):
            # A per-request length list repeats one length per sequence; a
            # kernel size or a padding is short enough to print as it stands.
            if len(set(value)) == 1 and len(value) > 3:
                return f"{value[0]}(×{len(value)})"
            return "[" + ",".join(str(x) for x in value[:4]) + ("…]" if len(value) > 4 else "]")
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _concrete_dtype(spec) -> str | None:
    """The dtype a signature pins for one input, when it pins exactly one."""
    if not isinstance(spec, dict):
        return None
    text = str(spec.get("dtype", ""))
    if not text or "|" in text or "(" in text:
        return None
    return text.strip() or None


# --- Resolution -------------------------------------------------------------

_SHAPE_SUFFIX = "_shape"


def _workload_of(entry: dict, config: str):
    """The manifest workload a benchmark id names, and the dtype it ran at."""
    for workload in entry.get("workloads") or []:
        label = workload.get("label")
        if not label:
            continue
        for dtype in workload.get("dtypes") or []:
            if config == f"{label}-{dtype}":
                return workload, label, dtype
    return None


def _derived_keys(workload: dict) -> set:
    """Keys whose value is the sum or the maximum of a list already shown.

    ``total_q`` is ``sum(q_lens)`` and ``max_seqlen_q`` is ``max(q_lens)``; a
    row that prints all three says one thing three times.
    """
    lists = [v for v in workload.values()
             if isinstance(v, list) and v and all(isinstance(x, int) for x in v)]
    if not lists:
        return set()
    derived = set()
    for key, value in workload.items():
        if not isinstance(value, int) or isinstance(value, bool):
            continue
        for seq in lists:
            if len(seq) > 1 and value in (sum(seq), max(seq)):
                derived.add(key)
                break
    return derived


def describe(entry: dict, config: str) -> Spec | None:
    """Describe one benchmarked workload, or None if the manifest lacks it."""
    found = _workload_of(entry, config)
    if not found:
        return None
    workload, label, dtype = found
    signature = entry.get("signature") or {}
    inputs = signature.get("inputs") or {}
    declared = signature.get("params") or {}

    scope = {k: v for k, v in workload.items()
             if isinstance(v, int) and not isinstance(v, bool)}
    scope.update({k.upper(): v for k, v in scope.items() if k.islower()})

    shapes: list[tuple[str, list, str | None]] = []  # (tensor, dims, dtype)
    consumed = set()

    for key, value in workload.items():
        if key.endswith(_SHAPE_SUFFIX) and isinstance(value, list):
            if not value:  # a 0-d scalar argument carries no shape
                consumed.add(key)
                continue
            name = key[: -len(_SHAPE_SUFFIX)]
            shapes.append((name, value, _concrete_dtype(inputs.get(name))))
            consumed.add(key)

    if not shapes and inputs:
        # No shape was given outright: evaluate what the signature templates.
        # All or nothing — an op whose templates cover three of its five inputs
        # would otherwise render a tensor list that is not the op's input list.
        templated, used = [], set()
        for name, spec in inputs.items():
            template = spec.get("shape") if isinstance(spec, dict) else None
            dims = _eval_template(str(template), scope) if template else None
            if dims is None:
                templated = []
                break
            templated.append((name, dims, _concrete_dtype(spec)))
            for symbol in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", str(template)):
                if symbol in workload:
                    used.add(symbol)
                elif symbol.lower() in workload:
                    used.add(symbol.lower())
        if templated:
            shapes, consumed = templated, consumed | used

    # Tensors of one shape and one dtype are named together on one line.
    grouped: "OrderedDict[tuple, list]" = OrderedDict()
    for name, dims, tensor_dtype in shapes:
        grouped.setdefault((fmt_shape(dims), tensor_dtype), []).append(name)
    tensors = [(" ".join(names), shape, tensor_dtype)
               for (shape, tensor_dtype), names in grouped.items()]

    derived = _derived_keys(workload)
    dims_out, params_out = [], []
    for key, value in workload.items():
        if key in consumed or key in _LABEL_KEYS or key in _DTYPE_KEYS:
            continue
        if key in derived:
            continue
        if key in declared:
            default = declared[key].get("default") if isinstance(declared[key], dict) else None
            if value != default:
                params_out.append((key, _fmt_value(value)))
            continue
        if isinstance(value, (int, float, str, list, bool)):
            dims_out.append((key, _fmt_value(value)))

    return Spec(label, dtype, tensors, dims_out, params_out)
