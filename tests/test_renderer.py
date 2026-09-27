"""What a fixed snapshot must render to, and the few rules worth stating twice.

The pages are the product, so the golden comparison carries most of the weight:
one fixed snapshot in, three committed pages out, byte for byte. A unit test
earns its place here only where the behaviour is a rule the pages do not show —
a template rejected, a package deciding where an op is published, an expression
the evaluator must refuse.
"""
import os
import shutil
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
    # No shapes to state, so every workload is named by its benchmark id alone,
    # the trailing dtype split off so the label's dtypes still share one group.
    assert "wl-tensor" not in "".join(pages.values())
    assert ('<td class="wl-name" rowspan="2"><code>decode-<wbr>b1-<wbr>h8</code></td>'
            '<td class="colsep">bf16</td>') in "".join(pages.values())


def test_a_manifest_in_a_subdirectory_renders_the_same_pages(tmp_path, rendered):
    # The YAML may sit in a subdirectory of the manifest package; the shapes
    # must not depend on which level it is read from.
    spec = tmp_path / "manifest" / "spec"
    spec.mkdir(parents=True)
    shutil.copy(os.path.join(FIXTURES, "manifest", "ops.yaml"), spec)
    pages, _ = render(str(tmp_path / "out"), manifest_dir=str(spec.parent))
    assert pages == rendered[0]


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


def test_the_api_reference_decides_the_op_order(tmp_path):
    # Pages in the order `nav` lists them, ops in the order a page names them,
    # and a path in a comment is not a nav entry.
    api = tmp_path / "api"
    api.mkdir()
    (api / "gemm.md").write_text("::: tileops.gemm.GemmFwdOp\n"
                                 "::: tileops.gemm.BmmFwdOp\n")
    (api / "elementwise.md").write_text("::: tileops.elementwise.AddFwdOp\n")
    yml = tmp_path / "mkdocs.yml"
    yml.write_text("# api/gemm.md is named here and is not a nav entry\n"
                   "nav:\n  - Elementwise: api/elementwise.md\n"
                   "  - GEMM: api/gemm.md\n")
    order = g.api_op_order(str(api), str(yml))
    assert list(order) == ["AddFwd", "GemmFwd", "BmmFwd"]

    # And the page follows it: the ops it names in that order, then the op it
    # names nowhere, whatever the verdicts say — `UnnamedFwd` leads by 9x and
    # still comes last, `GemmFwd` is behind and still comes first.
    def row(op, status, speedup):
        return (op, "tileops.ops.gemm.gemm", {"status": status,
                                              "speedup": speedup,
                                              "workloads": 0}, "", None)
    ops = ["BmmFwdOp", "GemmFwdOp", "UnnamedFwdOp"]
    page = g.data_page("GEMM", ["linear_algebra"],
                       {"linear_algebra": [row("BmmFwdOp", g.AHEAD, 4.0),
                                           row("GemmFwdOp", g.BEHIND, 0.5),
                                           row("UnnamedFwdOp", g.AHEAD, 9.0)]},
                       {op: [] for op in ops}, {op: [] for op in ops},
                       "main", order)
    assert [ln.split("[")[1].split("]")[0] for ln in page.splitlines()
            if ln.startswith("## [")] == ["GemmFwd", "BmmFwd", "UnnamedFwd"]


def test_the_package_decides_the_family_not_a_word_in_the_name():
    # `linear` matches `linear_attention` as a substring, so the package has to
    # win: otherwise a linear-attention op is published on the GEMM page.
    assert g.family_of("DeltaDecodeFwdOp",
                       "tileops.ops.linear_attention.delta") == "linear_attention"
    assert g.family_of("GemmOp", "tileops.ops.gemm.gemm") == "linear_algebra"
    # An op defined in a module rather than a package still falls through.
    assert g.family_of("RmsNormFwdOp", None) == "normalization"


def test_a_parametric_row_resolves_through_the_tileops_checkout():
    # The parametric format is instantiated by TileOPs itself, so this needs a
    # checkout: `./TileOPs`, or the one `TILEOPS` names.
    tileops = os.environ.get("TILEOPS", os.path.join(REPO, "TileOPs"))
    if not os.path.isdir(os.path.join(tileops, "src", "tileops", "manifest")):
        pytest.skip("no TileOPs checkout to instantiate parametric rows with")
    entry = {
        "family": "gemm", "status": "implemented",
        "signature": {
            "types": {"Mat": {"params": {"t": "Bool", "R": "Dim", "C": "Dim"},
                              "match": "t",
                              "cases": [{"when": False, "is": "[R, C]"},
                                        {"when": True, "is": "[C, R]"}]}},
            "forall": {"M": "Dim", "N": "Dim", "K": "Dim",
                       "T": "DType[float16 | bfloat16]"},
            "params": {"trans_b": {"type": "bool", "default": True}},
            "inputs": {"a": {"dtype": "T", "shape": "[M, K]"},
                       "b": {"dtype": "T", "shape": "Mat[trans_b, K, N]"}},
            "outputs": {"d": {"dtype": "T", "shape": "[M, N]"}}},
        "workloads": [{"M": 16, "N": 32, "K": 64, "trans_b": False,
                       "dtype_cases": [{"T": "bfloat16"}], "label": "nn"}],
        "roofline": {"flops": "2 * M * N * K"},
    }
    spec = ws.describe(entry, "nn-bfloat16", "GemmFwdOp", ws.Parametric(tileops, {}))
    # The type family is expanded on the branch the row selects, and the case
    # id is TileOPs' own: a dtype the row does not assign names no workload.
    assert spec.symbolic == [("a", "[M, K]", None), ("b", "[K, N]", None)]
    assert dict(spec.bindings) == {"M": 16, "K": 64, "N": 32}
    assert spec.params == [("trans_b", "false")]
    assert ws.describe(entry, "nn-float16", "GemmFwdOp", ws.Parametric(tileops, {})) is None
