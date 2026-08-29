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

    def __init__(self, label, dtype, tensors, dims, params,
                 symbolic=None, bindings=None):
        self.label = label          # "hidden-state-prefill"
        self.dtype = dtype          # "float16" — the workload's dtype row
        self.tensors = tensors      # [(names, "[2048, 4096]", dtype or None)]
        self.dims = dims            # [("heads", "32"), ...]
        self.params = params        # [("is_causal", "true"), ...]
        # The same tensors in the manifest's own symbols — [(names, "[B, H, DK]",
        # dtype)] — with what those symbols were set to on this workload.
        # None where the signature templates no shape, or where one symbol would
        # have to stand for two values.
        self.symbolic = symbolic
        self.bindings = bindings or OrderedDict()

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


_SYMBOL = re.compile(r"[A-Za-z_][A-Za-z_0-9]*$")


def _bind(template: str, dims: list) -> tuple[list, dict] | None:
    """Read a concrete shape back through the template that declares it.

    ``"[B, H, DK]"`` against ``[1, 8, 128]`` gives ``["B", "H", "DK"]`` and
    ``{B: 1, H: 8, DK: 128}``. A position holding an expression rather than a
    plain name keeps its number: ``[B * H, D]`` is not two symbols to solve for.
    """
    body = (template or "").strip()
    if not (body.startswith("[") and body.endswith("]")):
        return None
    parts = _split_dims(body[1:-1])
    if len(parts) != len(dims):
        return None
    symbols, binds = [], {}
    for part, value in zip(parts, dims, strict=True):
        if _SYMBOL.match(part):
            # One name twice in one shape means the two positions are equal.
            # `[B, B]` against `[1, 2]` is not that shape, so the template is
            # rejected rather than bound to whichever value came last.
            if binds.get(part, value) != value:
                return None
            symbols.append(part)
            binds[part] = value
        else:
            symbols.append(str(value))
    return symbols, binds


def _restated_by_symbol(entry: dict, bindings: dict, workload: dict) -> set:
    """Scalars a shape symbol already states, per the manifest's own rules.

    ``shape_rules`` carries the identities: ``B == batch`` means the row prints
    the same number twice, once as the first dimension of every tensor and once
    as a parameter. Only one-name identities count — ``S == num_chunks *
    chunk_len`` does not let a reader recover ``num_chunks`` from ``S``, so both
    stay.
    """
    rules = (entry.get("signature") or {}).get("shape_rules") or []
    out = set()
    for rule in rules:
        parts = str(rule).split("==")
        if len(parts) != 2:
            continue
        left, right = (p.strip() for p in parts)
        if not (_SYMBOL.match(left) and _SYMBOL.match(right)):
            continue
        for symbol, name in ((left, right), (right, left)):
            if symbol in bindings and workload.get(name) == bindings[symbol]:
                out.add(name)
    return out


def _symbolic(shapes: list, inputs: dict):
    """Every tensor in the manifest's symbols, or (None, None).

    All or nothing, and only where the symbols hold: one name standing for two
    values on the same workload means the template does not describe it, and
    printing the shapes in symbols would then say something untrue.
    """
    if not shapes or not inputs:
        return None, None
    out, binds = [], OrderedDict()
    for name, dims, tensor_dtype in shapes:
        spec = inputs.get(name)
        template = spec.get("shape") if isinstance(spec, dict) else None
        bound = _bind(str(template), dims) if template else None
        if bound is None:
            return None, None
        symbols, values = bound
        for symbol, value in values.items():
            if binds.setdefault(symbol, value) != value:
                return None, None
        out.append((name, "[" + ", ".join(symbols) + "]", tensor_dtype))
    return out, binds


# --- Formatting -------------------------------------------------------------


def fmt_shape(dims) -> str:
    return "[" + ", ".join(str(d) for d in dims) + "]"


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


def _int_lists(workload: dict) -> list:
    return [v for v in workload.values()
            if isinstance(v, list) and len(v) > 1 and all(isinstance(x, int) for x in v)]


def _restates(workload: dict, key: str):
    """Whether one key only restates lists the row already prints.

    ``total_q`` is ``sum(q_lens)``, ``max_seqlen_q`` is ``max(q_lens)``, and a
    paged run's ``max_position`` is ``max(cache_lens) + max(q_lens)``. Returns
    None where the key is absent or the row prints no list to check against, so
    a row that cannot settle the question does not vote either way.
    """
    value = workload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    lists = _int_lists(workload)
    if not lists:
        return None
    if any(value in (sum(seq), max(seq)) for seq in lists):
        return True
    maxes = [max(seq) for seq in lists]
    return any(value == a + b for i, a in enumerate(maxes) for b in maxes[i + 1:])


_DERIVED_CACHE = "_workload_shape_derived"


def _derived_keys(entry: dict) -> set:
    """Keys the whole op restates, across every workload that carries them.

    Judged per op rather than per row: one row where a sum happens to match a
    length is a coincidence, and dropping on it would hide a real dimension.
    A key is dropped only where every row that can settle it agrees, so a
    manifest that stops holding the identity brings the column straight back.
    """
    cached = entry.get(_DERIVED_CACHE)
    if cached is not None:
        return cached
    workloads = entry.get("workloads") or []
    candidates = {k for w in workloads for k, v in w.items()
                  if isinstance(v, int) and not isinstance(v, bool)}
    derived = set()
    for key in candidates:
        verdicts = [v for v in (_restates(w, key) for w in workloads) if v is not None]
        if verdicts and all(verdicts):
            derived.add(key)
    entry[_DERIVED_CACHE] = derived
    return derived


# A count and its key/value counterpart are one fact about the op's shape, read
# together everywhere the op is discussed: 32 query heads over 8 KV heads is
# `32/8`, not two entries a reader has to pair up.
_PAIRED_SUFFIX = "_kv"


def _pair_counts(items: list) -> list:
    """Fold ``heads 32 · heads_kv 8`` into ``heads 32/8``."""
    values = dict(items)
    folded, dropped = [], set()
    for key, value in items:
        if key in dropped:
            continue
        mate = key + _PAIRED_SUFFIX
        if mate in values:
            folded.append((key, f"{value}/{values[mate]}"))
            dropped.add(mate)
        else:
            folded.append((key, value))
    return [(k, v) for k, v in folded if k not in dropped]


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
    tensors = _group_tensors((name, fmt_shape(dims), dt) for name, dims, dt in shapes)
    sym_shapes, bindings = _symbolic(shapes, inputs)
    # Grouped by the symbolic shape, not the concrete one: `q: [B, H, DK]` and
    # `v: [B, H, DV]` are one line only on the workloads where DK == DV.
    symbolic = _group_tensors(sym_shapes) if sym_shapes else None

    derived = _derived_keys(entry) | _restated_by_symbol(entry, bindings or {},
                                                        workload)
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

    return Spec(label, dtype, tensors, _pair_counts(dims_out), params_out,
                symbolic, bindings)


def _group_tensors(shapes) -> list:
    """Tensors of one shape and one dtype, named together on one line."""
    grouped: OrderedDict[tuple, list] = OrderedDict()
    for name, shape, tensor_dtype in shapes:
        grouped.setdefault((shape, tensor_dtype), []).append(name)
    # Comma-separated: `q, k` is two tensors of one shape, and a space alone
    # does not say that at a glance.
    return [(", ".join(names), shape, tensor_dtype)
            for (shape, tensor_dtype), names in grouped.items()]
