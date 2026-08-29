"""Fixtures shared by the renderer's tests.

`scripts/` is not a package — the deploy runs its modules as scripts — so the
directory is put on the path here rather than imported from an installed name.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

FIXTURES = os.path.join(REPO, "tests", "fixtures")
GOLDEN = os.path.join(REPO, "tests", "golden")


@pytest.fixture
def fixtures() -> str:
    return FIXTURES


@pytest.fixture
def manifest() -> dict:
    import workload_shape
    return workload_shape.load_manifest(os.path.join(FIXTURES, "manifest"))
