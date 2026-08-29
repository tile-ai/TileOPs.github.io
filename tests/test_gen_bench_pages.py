"""What the renderer reads out of a snapshot, and what verdict it reaches."""
import os

import gen_bench_pages as g


def _workloads(fixtures):
    return g.parse_bench_xml(os.path.join(fixtures, "bench_results.xml"))


# --- Reading the snapshot ---------------------------------------------------

def test_parse_splits_passing_failing_and_skipped(fixtures):
    workloads, failures, skips = _workloads(fixtures)
    assert [w["config"] for w in workloads] == [
        "decode-b1-h8-bfloat16", "decode-b8-h8-bfloat16", "scan-b2-bfloat16",
        "scan-b4-bfloat16", "undeclared-op-case-float16", "oblong-float16"]
    assert [f["op"] for f in failures] == ["MysteryFwdOp"]
    assert [s["op"] for s in skips] == ["ChunkScanFwdOp"]


def test_parse_groups_properties_by_implementation(fixtures):
    first = _workloads(fixtures)[0][0]
    assert set(first["impls"]) == {"tileops", "fla", "torch-ref"}
    assert first["impls"]["tileops"]["device_busy_ms"] == 0.0031
    assert first["impls"]["fla"]["ratio"] == 4.58
    # Non-numeric metrics stay strings; numeric ones are parsed.
    assert first["impls"]["tileops"]["timing"] == "cupti"


def test_correctness_tally_keeps_the_worst_error(fixtures):
    tests = g.parse_test_xml(os.path.join(fixtures, "test_results.xml"))
    assert tests["DeltaDecodeFwdOp"]["passed"] == 1
    assert tests["DeltaDecodeFwdOp"]["max_abs_err"] == 0.0009765625
    assert tests["ChunkScanFwdOp"]["failed"] == 1
    assert tests["MysteryFwdOp"]["skipped"] == 1


# --- Baselines and verdicts -------------------------------------------------

def test_a_reference_tag_is_a_reference_whatever_its_prefix():
    assert g.tier_of("torch-ref") == g.TIER_REF
    assert g.tier_of("some_ref") == g.TIER_REF


def test_torch_native_and_library_tiers():
    assert g.tier_of("torch") == g.TIER_TORCH
    assert g.tier_of("fla") == g.TIER_LIB


def test_an_unknown_tag_is_reported_rather_than_silently_rated(fixtures):
    # Unknown tags are rendered as library kernels, so the run has to say which
    # ones it did not recognise — `triton` is known, a made-up one is not.
    workloads, _, _ = _workloads(fixtures)
    tags = {t for w in workloads for t in w["impls"] if not t.startswith("tileops")}
    assert tags - g._KNOWN_TAGS == set()


def test_rated_against_the_fastest_real_alternative(fixtures):
    workloads, _, _ = _workloads(fixtures)
    m = g.workload_metrics(workloads[0])
    assert m["busy_ms"] == 0.0031
    assert m["rivals"]["fla"]["tier"] == g.TIER_LIB
    assert m["rivals"]["fla"]["speedup"] == 4.58


def test_an_op_with_only_an_eager_reference_stays_unrated(fixtures):
    workloads, _, _ = _workloads(fixtures)
    scan = [w for w in workloads if w["config"] == "scan-b2-bfloat16"]
    s = g.op_summary([g.workload_metrics(w) for w in scan])
    assert s["status"] == g.UNRATED
    assert s["rival_ref_only"] is True


def test_an_op_measured_against_a_library_kernel_is_rated(fixtures):
    workloads, _, _ = _workloads(fixtures)
    delta = [w for w in workloads if w["op"] == "DeltaDecodeFwdOp"]
    s = g.op_summary([g.workload_metrics(w) for w in delta])
    # 4.58x on one workload and 0.5x on the other: the section is ordered by
    # the geometric mean of the two, 1.51x, and rated because the rival is a
    # real kernel rather than an eager reference.
    assert s["rival"] == "fla"
    assert round(s["speedup"], 2) == 1.51
    assert s["status"] == g.AHEAD
    assert s["workloads"] == 2


def test_recorded_and_computed_ratios_are_both_kept(fixtures):
    workloads, _, _ = _workloads(fixtures)
    scan = [w for w in workloads if w["config"] == "scan-b4-bfloat16"][0]
    m = g.workload_metrics(scan)
    r = m["rivals"]["torch"]
    assert r["recorded_ratio"] == 1.6      # what the benchmark wrote
    assert r["computed_ratio"] == 1.0      # what the published times say


def test_a_ratio_that_disagrees_with_the_published_times_is_collected(fixtures):
    workloads, _, _ = _workloads(fixtures)
    scan = [w for w in workloads if w["op"] == "ChunkScanFwdOp"]
    drift = g.collect_ratio_drift(scan, [g.workload_metrics(w) for w in scan])
    assert [(tag, rec, comp) for _, tag, rec, comp in drift] == [("torch", 1.6, 1.0)]


# --- Where an op is published ----------------------------------------------

def test_the_package_decides_the_family_not_a_word_in_its_name():
    # `linear` matches `linear_attention` as a substring, so the package has to
    # win: otherwise a linear-attention op is published on the GEMM page.
    assert g.family_of("DeltaDecodeFwdOp",
                       "tileops.ops.linear_attention.delta") == "linear_attention"
    assert g.family_of("GemmOp", "tileops.ops.gemm.gemm") == "linear_algebra"
    assert g.family_of("ChunkScanFwdOp", "tileops.ops.mamba.scan") == "linear_attention"


def test_an_op_with_no_module_is_classified_by_its_name():
    assert g.family_of("RmsNormFwdOp", None) == "normalization"


def test_every_family_lands_on_a_page():
    for fam in g.FAMILY_TITLE:
        assert g.page_of_family(fam) in {slug for slug, _, _ in g.DATA_PAGES}
