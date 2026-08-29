"""What a fixed snapshot must render to, and the few rules worth stating twice.

The pages are the product, so the golden comparison carries most of the weight:
one fixed snapshot in, three committed pages out, byte for byte. A unit test
earns its place here only where the behaviour is a rule the pages do not show —
a template rejected, a package deciding where an op is published, an expression
the evaluator must refuse.
"""
import os
import subprocess
import sys

import gen_bench_pages as g
import pytest
import workload_shape as ws

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(REPO, "tests", "fixtures")
GOLDEN = os.path.join(REPO, "tests", "golden")

# The method note is prose written in the renderer, with nothing read out of the
# snapshot: holding it byte for byte would only mean refreshing a golden file
# whenever a sentence is edited, and the sentence is already in the diff.
PROSE = {"reading.md"}


def render(out_dir: str, manifest_dir: str = os.path.join(FIXTURES, "manifest")):
    """Run the renderer as the deploy runs it, and return what it wrote.

    The roofline tool is pointed at a directory that holds nothing, so the SOL
    column is the degraded one and the pages depend on this repository alone: a
    contributor with a TileOPs checkout beside them renders what CI renders.
    """
    cmd = [sys.executable, os.path.join(REPO, "scripts", "gen_bench_pages.py"),
           "--tileops", os.path.join(FIXTURES, "no-tileops"),
           "--bench-xml", os.path.join(FIXTURES, "bench_results.xml"),
           "--test-xml", os.path.join(FIXTURES, "test_results.xml"),
           "--meta", os.path.join(FIXTURES, "meta.json"),
           "--manifest-dir", manifest_dir,
           "--commit", "0123456789abcdef0123456789abcdef01234567",
           "--date", "2026-01-01", "--gpu", "NVIDIA H200", "--out-dir", out_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return {n: open(os.path.join(out_dir, n), encoding="utf-8").read()
            for n in sorted(os.listdir(out_dir))}, proc.stderr


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    return render(str(tmp_path_factory.mktemp("bench")))


# --- The pages --------------------------------------------------------------

def test_pages_match_the_committed_output(rendered):
    pages = {n: t for n, t in rendered[0].items() if n not in PROSE}
    assert sorted(pages) == sorted(os.listdir(GOLDEN))
    for name, text in pages.items():
        expected = open(os.path.join(GOLDEN, name), encoding="utf-8").read()
        assert text == expected, (
            f"{name} changed. Read the diff — it is the change, stated in the "
            f"product — then run: python tests/refresh_golden.py")


def test_rendering_is_deterministic(tmp_path):
    # Two runs over one snapshot, so an ordering that depends on a set or on
    # dict iteration shows up here rather than as a diff on the deployed site.
    assert render(str(tmp_path / "a"))[0] == render(str(tmp_path / "b"))[0]


def test_a_run_reports_what_it_could_not_describe(rendered):
    stderr = rendered[1]
    assert "MysteryFwdOp" in stderr              # no manifest entry
    assert "recorded ratio disagrees" in stderr  # ratio against the times
    assert "brand-new-lib" in stderr             # a baseline tag with no tier


def test_without_a_manifest_the_pages_still_render(tmp_path):
    pages, _ = render(str(tmp_path / "bare"), manifest_dir=str(tmp_path / "none"))
    assert pages, "a missing manifest must not stop the deploy"
    # No shapes to state, so every workload is named by its benchmark id alone.
    assert "wl-tensor" not in "".join(pages.values())
    assert "decode-b1-h8-bfloat16" in "".join(pages.values())


# --- Rules the pages do not show -------------------------------------------

def test_a_template_may_not_bind_one_symbol_to_two_values():
    # `[D, D]` says the two dimensions are equal. This shape says they are not,
    # so the template is rejected and the row prints its concrete shape.
    assert ws._bind("[D, D]", [64, 32]) is None
    assert ws._bind("[D, D]", [64, 64]) == (["D", "D"], {"D": 64})


def test_a_template_is_not_executed():
    # Templates are parsed, not run: only integer arithmetic over the names a
    # workload sets resolves, and anything else leaves the tensor undescribed.
    assert ws._eval_template("[max(a, b)]", {"a": 1, "b": 2}) is None
    assert ws._eval_template("[n * 2, k]", {"n": 4, "k": 3}) == [8, 3]


def test_the_package_decides_the_family_not_a_word_in_the_name():
    # `linear` matches `linear_attention` as a substring, so the package has to
    # win: otherwise a linear-attention op is published on the GEMM page.
    assert g.family_of("DeltaDecodeFwdOp",
                       "tileops.ops.linear_attention.delta") == "linear_attention"
    assert g.family_of("GemmOp", "tileops.ops.gemm.gemm") == "linear_algebra"
    # An op defined in a module rather than a package still falls through.
    assert g.family_of("RmsNormFwdOp", None) == "normalization"
