"""What a benchmarked workload resolves to, given the manifest that declares it."""
import workload_shape as ws


def spec(manifest, op, config):
    return ws.describe(manifest[op], config)


# --- Templates and their symbols -------------------------------------------

def test_eval_template_resolves_names_from_the_workload():
    assert ws._eval_template("[batch, seq]", {"batch": 2, "seq": 8}) == [2, 8]


def test_eval_template_does_arithmetic_over_those_names():
    assert ws._eval_template("[n * 2, k]", {"n": 4, "k": 3}) == [8, 3]


def test_eval_template_refuses_a_name_the_workload_does_not_set():
    assert ws._eval_template("[batch, heads]", {"batch": 2}) is None


def test_eval_template_refuses_a_call():
    assert ws._eval_template("[max(a, b)]", {"a": 1, "b": 2}) is None


def test_bind_reads_a_concrete_shape_back_through_its_template():
    symbols, binds = ws._bind("[B, H, DK]", [1, 8, 128])
    assert symbols == ["B", "H", "DK"]
    assert binds == {"B": 1, "H": 8, "DK": 128}


def test_bind_keeps_the_number_where_the_template_holds_an_expression():
    symbols, binds = ws._bind("[B * H, D]", [8, 64])
    assert symbols == ["8", "D"]
    assert binds == {"D": 64}


def test_bind_rejects_one_name_standing_for_two_values():
    # `[D, D]` says the two dimensions are equal. This shape says they are not,
    # so the template does not describe it and no binding may be invented.
    assert ws._bind("[D, D]", [64, 32]) is None


def test_bind_accepts_a_repeated_name_where_the_values_agree():
    symbols, binds = ws._bind("[D, D]", [64, 64])
    assert symbols == ["D", "D"] and binds == {"D": 64}


def test_bind_rejects_a_template_of_another_rank():
    assert ws._bind("[B, H]", [1, 8, 128]) is None


# --- Whole workloads --------------------------------------------------------

def test_shapes_given_outright_are_described_in_the_signature_symbols(manifest):
    s = spec(manifest, "DeltaDecodeFwdOp", "decode-b1-h8-bfloat16")
    # A tensor whose signature dtype is a choice or `same_as(q)` carries no
    # dtype of its own: it was measured in the workload's, and the row says so
    # once. Only a tensor pinned to one dtype names it here.
    assert s.tensors == [("q, k, v", "[1, 8, 128]", None),
                         ("state", "[1, 8, 128, 128]", None)]
    assert s.symbolic == [("q, k", "[B, H, DK]", None),
                          ("v", "[B, H, DV]", None),
                          ("state", "[B, H, DK, DV]", None)]
    assert s.bindings == {"B": 1, "H": 8, "DK": 128, "DV": 128}


def test_tensors_of_one_concrete_shape_are_named_together(manifest):
    # q, k and v are one line by shape; only the symbols tell DK from DV.
    s = spec(manifest, "DeltaDecodeFwdOp", "decode-b8-h8-bfloat16")
    assert s.tensors[0][0] == "q, k, v"
    assert s.bindings["B"] == 8


def test_a_template_no_symbol_can_describe_falls_back_to_concrete(manifest):
    s = spec(manifest, "SquareFwdOp", "oblong-float16")
    assert s.symbolic is None
    assert s.tensors == [("a", "[64, 32]", "float16")]  # pinned in the signature


def test_scalars_the_manifest_names_are_reported(manifest):
    s = spec(manifest, "ChunkScanFwdOp", "scan-b2-bfloat16")
    assert ("num_chunks", "4") in s.dims
    assert ("chunk_len", "64") in s.dims


def test_a_parameter_at_its_default_is_not_reported(manifest):
    # is_causal defaults to true in the signature, so the row that takes the
    # default says nothing and the row that does not says so.
    at_default = spec(manifest, "ChunkScanFwdOp", "scan-b2-bfloat16")
    changed = spec(manifest, "ChunkScanFwdOp", "scan-b4-bfloat16")
    assert at_default.params == []
    assert changed.params == [("is_causal", "false")]


def test_an_undeclared_workload_resolves_to_nothing(manifest):
    assert ws.describe(manifest["DeltaDecodeFwdOp"], "no-such-label-bfloat16") is None


def test_dtype_abbreviations():
    assert ws.abbr_dtype("bfloat16") == "bf16"
    assert ws.abbr_dtype("float8_e4m3fn") == "fp8e4m3"
    assert ws.abbr_dtype("something_else") == "something_else"


def test_fmt_shape_is_a_bracketed_list():
    assert ws.fmt_shape([1, 8, 128]) == "[1, 8, 128]"
