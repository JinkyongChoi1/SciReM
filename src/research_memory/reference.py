"""Deterministic reference audit; expose gold targets only after discovery is frozen."""

import math


def compare_reference(actual, expected, relative_tolerance=1e-5, absolute_tolerance=1e-8):
    if not isinstance(actual, dict) or not isinstance(expected, dict) or not expected:
        raise ValueError("Actual and reference artifacts must be nonempty target mappings")
    if not all(isinstance(t, (int, float)) and math.isfinite(t) and t >= 0 for t in (relative_tolerance, absolute_tolerance)):
        raise ValueError("Tolerances must be finite nonnegative values")
    def equal(found, gold):
        if isinstance(gold, dict):
            return isinstance(found, dict) and all(key in found and equal(found[key], value) for key, value in gold.items())
        if isinstance(gold, list):
            return isinstance(found, list) and len(found) == len(gold) and all(equal(a, b) for a, b in zip(found, gold))
        if type(gold) is int:
            return type(found) is int and found == gold
        if type(gold) is float:
            return type(found) in (int, float) and math.isfinite(found) and math.isfinite(gold) and math.isclose(found, gold, rel_tol=relative_tolerance, abs_tol=absolute_tolerance)
        return type(found) is type(gold) and found == gold
    return {target: target in actual and equal(actual[target], value) for target, value in expected.items()}
