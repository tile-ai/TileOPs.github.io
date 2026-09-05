#!/usr/bin/env python3
"""Compare the ops the API pages name against the ops a TileOPs checkout exports.

The pages under `docs/api/` name their ops one by one, and every build checks
TileOPs out fresh. An op the pages name and the checkout no longer exports fails
this check: mkdocs would otherwise abort mid-build on `Could not collect`. An op
exported with no page is printed and does not fail — TileOPs adds ops on its own
schedule, and an unrelated pull request here is not the place to stop for one.

Families come from `_FAMILIES` in `tileops/__init__.py`, their ops from the
`__all__` of `tileops/<family>.py`, read with `ast`: importing a family pulls in
torch, which the docs environment does not install. A path of more than two
segments, such as `tileops.trace.api._Trace`, belongs to no family and is left
alone.

Usage:
    python scripts/check_api_pages.py [--tileops TileOPs/src] [--docs docs/api]
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

IDENTIFIER = re.compile(r"^\s*::: +tileops\.(\w+)\.(\w+)\s*$", re.MULTILINE)


def _names(path: Path, variable: str) -> list[str]:
    """The strings assigned to `variable` at the top level of the module at `path`."""
    if not path.is_file():
        raise SystemExit(f"no such file: {path} — is there a TileOPs checkout?")
    for node in ast.parse(path.read_text(encoding="utf-8"), filename=str(path)).body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == variable for t in node.targets):
            continue
        # Anything but a literal of strings is refused rather than read past: a
        # name silently dropped here is an op this check would stop looking at.
        if not isinstance(node.value, ast.Tuple | ast.List):
            raise SystemExit(f"{path}: {variable} is not a list or tuple literal")
        names = [
            e.value
            for e in node.value.elts
            if isinstance(e, ast.Constant) and isinstance(e.value, str)
        ]
        if len(names) != len(node.value.elts):
            raise SystemExit(f"{path}: {variable} holds something other than plain strings")
        return names
    raise SystemExit(f"{path}: no top-level {variable}")


def exported(src: Path) -> dict[str, list[str]]:
    """Op names per family, from the TileOPs source tree at `src`."""
    families = _names(src / "tileops" / "__init__.py", "_FAMILIES")
    return {f: _names(src / "tileops" / f"{f}.py", "__all__") for f in families}


def documented(pages: Path) -> dict[str, dict[str, str]]:
    """Op names per family, each mapped to the page that names it."""
    found: dict[str, dict[str, str]] = {}
    for page in sorted(pages.glob("*.md")):
        for family, op in IDENTIFIER.findall(page.read_text(encoding="utf-8")):
            found.setdefault(family, {})[op] = page.name
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tileops", type=Path, default=Path("TileOPs/src"))
    parser.add_argument("--docs", type=Path, default=Path("docs/api"))
    args = parser.parse_args()

    families = exported(args.tileops)
    pages = documented(args.docs)
    # Empty means the pages carry no identifier at all — a regex that stopped
    # matching, not a page naming an op wrongly, which is reported below.
    if not pages:
        raise SystemExit(f"no `::: tileops.<family>.<Op>` identifier under {args.docs}")

    uncollectable, undocumented = [], []
    for family, on_page in sorted(pages.items()):
        if family not in families:
            uncollectable += [
                f"{page}: tileops.{family}.{op} names no op family of the checkout"
                for op, page in sorted(on_page.items())
            ]
            continue
        uncollectable += [
            f"{on_page[op]}: tileops.{family}.{op} is not in the `__all__` of "
            f"tileops/{family}.py — the build cannot collect it"
            for op in sorted(set(on_page) - set(families[family]))
        ]
    for family, ops in families.items():
        undocumented += [
            f"tileops.{family}.{op} is exported, and no page under {args.docs} names it"
            for op in ops
            if op not in pages.get(family, {})
        ]

    for line in uncollectable + undocumented:
        print(line, file=sys.stderr)
    if uncollectable:
        print(f"\n{len(uncollectable)} op(s) the build cannot collect", file=sys.stderr)
        return 1
    if undocumented:
        print(f"{len(undocumented)} exported op(s) no page names")
        return 0
    print(f"{sum(len(ops) for ops in families.values())} ops, on the page and exported")
    return 0


if __name__ == "__main__":
    sys.exit(main())
