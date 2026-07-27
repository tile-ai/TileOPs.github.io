"""Regression tests for benchmark family classification."""

import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location(
    "gen_bench_pages", SCRIPTS / "gen_bench_pages.py"
)
assert spec and spec.loader
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

from benchmark_families import FAMILY_OPS, OP_FAMILY  # noqa: E402


class BenchmarkFamilyTests(unittest.TestCase):

    def test_reviewed_entries_are_unique_and_renderable(self):
        declared_count = sum(len(ops) for ops in FAMILY_OPS.values())
        self.assertEqual(len(OP_FAMILY), declared_count)
        self.assertLessEqual(set(FAMILY_OPS), set(gen.FAMILY_TITLE))

    def test_reviewed_family_corrects_inference(self):
        cases = [
            (
                "MeanPoolingForwardOp",
                "tileops.ops.attention.deepseek_nsa",
                "attention",
                "pool",
            ),
            (
                "EngramDecodeOp",
                "tileops.ops.engram_decode",
                "linear_attention",
                "sequence_modeling",
            ),
            ("BmmFp8Op", "tileops.ops.bmm", "quantization", "linear_algebra"),
            (
                "FP8LightningIndexerOp",
                "tileops.ops.fp8_lightning_indexer",
                "quantization",
                "attention_indexing",
            ),
            ("maximum", None, "reduction", "elementwise"),
        ]
        for op, module, inferred, reviewed in cases:
            with self.subTest(op=op):
                self.assertEqual(gen.infer_family(op, module), inferred)
                self.assertEqual(gen.family_of(op, module), reviewed)

    def test_unknown_operator_keeps_inferred_family(self):
        self.assertEqual(
            gen.family_of("FutureGemmOp", None),
            "linear_algebra",
        )


if __name__ == "__main__":
    unittest.main()
