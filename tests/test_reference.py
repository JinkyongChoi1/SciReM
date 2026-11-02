import unittest

from research_memory.reference import compare_reference


class ReferenceAuditTests(unittest.TestCase):
    def test_coefficients_tolerance_and_exact_sample(self):
        expected = {"baseline": {"n": 100, "coefficient": .12345, "stderr": .05, "sample_hash": "same"}}
        actual = {"baseline": {"n": 100, "coefficient": .12345001, "stderr": .05, "sample_hash": "same"}}
        self.assertTrue(compare_reference(actual, expected)["baseline"])
        actual["baseline"]["n"] = 101
        self.assertFalse(compare_reference(actual, expected)["baseline"])

    def test_missing_or_nonfinite_result_is_not_faithful(self):
        self.assertFalse(compare_reference({}, {"coefficient": .1})["coefficient"])
        self.assertFalse(compare_reference({"coefficient": float("nan")}, {"coefficient": .1})["coefficient"])
        self.assertFalse(compare_reference({"n": True}, {"n": 1})["n"])

    def test_invalid_tolerance_rejected(self):
        with self.assertRaises(ValueError):
            compare_reference({"a": 1}, {"a": 1}, absolute_tolerance=-1)
