"""CIAT's 40 response cases (19 numeric + 21 host gate) and 6 policy cases."""
import math

import pytest

from nbs_runtime.aggregate import combine
from nbs_runtime.gates import evaluate_gate
from nbs_runtime.responses import evaluate_option


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, abs_tol=1e-9)


def _cases(name, kind):
    import json

    from nbs_runtime.contract import load_package

    p = load_package()
    data = json.loads((p.root / name).read_text(encoding="utf-8"))
    if kind == "numeric":
        return [c for c in data if "case" not in c]
    return [c for c in data if "case" in c]


@pytest.mark.parametrize("c", _cases("conformance_examples.json", "numeric"),
                         ids=lambda c: f"{c['rule_id'].split('__', 1)[-1]}|{c['function_id']}")
def test_numeric_case(pkg, c):
    out = evaluate_option(pkg.option(c["rule_id"], c["function_id"]), c["input"])
    assert all(_same(a, b) for a, b in zip(out, c["expected"], strict=True)), (out, c["expected"])


@pytest.mark.parametrize("c", _cases("conformance_examples.json", "gate"),
                         ids=lambda c: f"{c['rule_id'].split('__', 1)[-1]}|{c['case']}")
def test_gate_case(pkg, c):
    params = pkg.rows[c["rule_id"]]["declared_relationship"]["params"]
    values, statuses = evaluate_gate(params, c["input"], c.get("host_masks"))
    assert all(_same(a, b) for a, b in zip(values, c["expected"], strict=True)), (values, c["expected"])
    assert statuses == c["expected_status"]


def test_counts(conformance):
    assert len(conformance) == 40


def test_policy_cases(pkg, policy_cases):
    assert len(policy_cases) == 6
    for c in policy_cases:
        policy = pkg.policy(c["policy_id"])
        res = combine(policy, c["scores"], c["host"], c["excluded"], c["eligible_fraction"])
        exp = c["expected"]
        assert res["status"] == exp["status"], c
        assert _same(res["score"], exp["score"]), c
        assert _same(res["minimum"], exp["minimum"]), c
        assert _same(res["eligible_fraction"], exp["eligible_fraction"]), c
