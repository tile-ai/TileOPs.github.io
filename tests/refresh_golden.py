#!/usr/bin/env python3
"""Rewrite tests/golden/ from the fixture snapshot.

Run after a deliberate change to what the pages say, and read the diff: it is
the change, stated in the product rather than in the code.
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_render_e2e import GOLDEN, render  # noqa: E402

if __name__ == "__main__":
    tmp = os.path.join(os.path.dirname(GOLDEN), ".golden-tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    pages, _ = render(tmp)
    shutil.rmtree(GOLDEN, ignore_errors=True)
    os.makedirs(GOLDEN)
    for name, text in pages.items():
        with open(os.path.join(GOLDEN, name), "w", encoding="utf-8") as f:
            f.write(text)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {len(pages)} pages to {GOLDEN}")
