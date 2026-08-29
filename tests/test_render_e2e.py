"""The whole renderer over a fixed snapshot, against committed output.

The pages are the product; every rule in the renderer is a rule about what they
say. A golden comparison is what makes a change to any of it visible in review
rather than on the published site.
"""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(REPO, "tests", "fixtures")
GOLDEN = os.path.join(REPO, "tests", "golden")
COMMIT = "0123456789abcdef0123456789abcdef01234567"


def render(out_dir: str, manifest_dir: str = os.path.join(FIXTURES, "manifest")):
    """Run the renderer as the deploy runs it, and return what it wrote.

    The roofline tool is pointed at a directory that holds nothing: the SOL
    column is then the degraded one, and the pages depend on this repository
    alone. A machine with a TileOPs checkout beside it renders what CI renders.
    """
    cmd = [sys.executable, os.path.join(REPO, "scripts", "gen_bench_pages.py"),
           "--tileops", os.path.join(FIXTURES, "no-tileops"),
           "--bench-xml", os.path.join(FIXTURES, "bench_results.xml"),
           "--test-xml", os.path.join(FIXTURES, "test_results.xml"),
           "--meta", os.path.join(FIXTURES, "meta.json"),
           "--manifest-dir", manifest_dir,
           "--commit", COMMIT, "--date", "2026-01-01",
           "--gpu", "NVIDIA H200", "--out-dir", out_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return {n: open(os.path.join(out_dir, n), encoding="utf-8").read()
            for n in sorted(os.listdir(out_dir))}, proc.stderr


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    return render(str(tmp_path_factory.mktemp("bench")))


def test_pages_match_the_committed_output(rendered):
    pages, _ = rendered
    assert sorted(pages) == sorted(os.listdir(GOLDEN))
    for name, text in pages.items():
        expected = open(os.path.join(GOLDEN, name), encoding="utf-8").read()
        assert text == expected, (
            f"{name} changed. If the change is intended, refresh the golden "
            f"files with: python tests/refresh_golden.py")


def test_rendering_is_deterministic(tmp_path):
    # Two runs over one snapshot, so an ordering that depends on a set or on
    # dict iteration shows up here rather than as a diff on the deployed site.
    first, _ = render(str(tmp_path / "a"))
    second, _ = render(str(tmp_path / "b"))
    assert first == second


def test_a_run_reports_what_it_could_not_describe(rendered):
    _, stderr = rendered
    # One op is absent from the manifest fragment, and one recorded ratio
    # disagrees with the times published beside it. Both must be said out loud.
    assert "MysteryFwdOp" in stderr
    assert "recorded ratio disagrees" in stderr


def test_without_a_manifest_the_pages_still_render(tmp_path):
    pages, _ = render(str(tmp_path / "bare"), manifest_dir=str(tmp_path / "none"))
    assert pages, "a missing manifest must not stop the deploy"
    # No shapes to state, so every workload is named by its benchmark id alone.
    assert "wl-tensor" not in "".join(pages.values())
    assert "decode-b1-h8-bfloat16" in "".join(pages.values())
