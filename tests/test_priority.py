import json
from copy import deepcopy
from pathlib import Path

import pytest

from nbs_runtime.priority import classify, contract, critic_weights, fit_rule, run


@pytest.fixture
def payload():
    return json.loads((Path(__file__).parents[1] / "config/priority/illustration.json").read_text())


def test_example_contributions_unknown_and_mask(payload):
    result = run(**payload)
    assert [p["status"] for p in result["pixels"]] == [
        "scored", "scored", "unknown", "excluded", "unknown"]
    assert result["pixels"][0]["score"] == pytest.approx(0.8325)
    assert result["pixels"][1]["score"] == pytest.approx(0.745)
    assert result["weights"] == {"poverty": 0.75, "flood": 0.25}
    assert sum(result["pixels"][0]["contributions"].values()) == result["pixels"][0]["score"]
    assert len(result["inputs"]["poverty"]["values_sha256"]) == 64
    assert result == run(**payload)


def test_exact_endpoints_and_direction(payload):
    rule = payload["config"]["rules"][0]
    fitted = fit_rule(rule)
    values = [-1, 0, 0.01, 20, 20.01, 40, 40.01, None]
    assert [classify(v, fitted) for v in values] == [0, 0, .33, .33, .66, .66, 1, None]
    rule["direction"] = "lower_is_more_concern"
    assert [classify(v, fit_rule(rule)) for v in values] == [1, 1, .66, .66, .33, .33, 0, None]


def test_zero_weight_does_not_require_missing_layer(payload):
    payload["config"]["weights"]["poverty"] = 0
    del payload["layers"]["T5:poverty_example"]
    result = run(**payload)
    assert result["pixels"][2]["score"] == 1
    assert result["weights"] == {"flood": 1}


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf"), True])
def test_invalid_weights(payload, weight):
    payload["config"]["weights"]["poverty"] = weight
    with pytest.raises(ValueError):
        run(**payload)


def test_large_weights_normalize_without_overflow(payload):
    payload["config"]["weights"] = {"poverty": 1e308, "flood": 1e308}
    assert run(**payload)["weights"] == {"poverty": .5, "flood": .5}


def test_zero_weights_rejected(payload):
    payload["config"]["weights"] = {"poverty": 0, "flood": 0}
    with pytest.raises(ValueError, match="All weights"):
        run(**payload)


def test_unit_and_alignment_validation(payload):
    bad = deepcopy(payload)
    bad["layers"]["T2:flood_example"]["unit"] = "mm"
    with pytest.raises(ValueError, match="unit mismatch"):
        run(**bad)
    payload["layers"]["T2:flood_example"]["values"].pop()
    with pytest.raises(ValueError, match="length mismatch"):
        run(**payload)


@pytest.mark.parametrize("cuts", [[0, 0, 2], [2, 1, 0], [0, 1], [0, 1, float("inf")]])
def test_bad_cutoffs(payload, cuts):
    payload["config"]["rules"][0]["cuts"] = cuts
    with pytest.raises(ValueError):
        run(**payload)


def test_alias_cannot_be_counted_as_another_concern(payload):
    alias = dict(payload["config"]["variables"][1], table="T5")
    payload["config"]["variables"].append(alias)
    payload["config"]["selection"]["flood_again"] = {
        "variable": "T5:flood_example", "reason": "Duplicate alias"}
    payload["config"]["weights"]["flood_again"] = 1
    with pytest.raises(ValueError, match="representative"):
        run(**payload)


@pytest.mark.parametrize("role", ["descriptor", "investment_risk"])
def test_nonpriority_cannot_enter_score(payload, role):
    payload["config"]["variables"][1]["mcda_role"] = role
    with pytest.raises(ValueError, match="Only priority"):
        run(**payload)


def test_missing_classification_has_no_historical_fallback(payload):
    payload["config"]["rules"] = []
    with pytest.raises(ValueError, match="Missing classification"):
        run(**payload)


def test_illustration_cannot_claim_country_analysis(payload):
    payload["config"]["purpose"] = "analysis"
    with pytest.raises(ValueError, match="Illustrative"):
        run(**payload)


def test_missing_and_nan_are_unknown_not_none(payload):
    payload["layers"]["T5:poverty_example"]["values"][0] = float("nan")
    assert run(**payload)["pixels"][0]["status"] == "unknown"


def test_empty_opportunity_and_constant_layers(payload):
    payload["opportunity"] = [False] * 5
    assert run(**payload)["counts"] == {"excluded": 5}
    payload["opportunity"] = [True] * 5
    for layer in payload["layers"].values():
        layer["values"] = [0] * 5
    result = run(**payload)
    assert all(p["score"] == 0 for p in result["pixels"])
    assert result["correlations"]["classified"]["pearson"]["poverty"]["flood"] is None


def test_terciles_explicit_none_and_saved_reference(payload):
    rule = payload["config"]["rules"][0]
    del rule["cuts"]
    rule.update(method="terciles", reference={"population": "synthetic six cells", "period": "baseline"},
                none_condition={"op": "le", "value": 0})
    fitted = fit_rule(rule, [None, 0, 1, 2, 3, 4, 5, 6])
    assert fitted["resolved_cuts"] == pytest.approx([8 / 3, 13 / 3])
    assert fitted["fitted_count"] == 6
    assert [classify(v, fitted) for v in [0, 1, 3, 6, 60]] == [0, .33, .66, 1, 1]
    assert fitted["reference_sha256"] != fit_rule(rule, [1, 2, 3, 4, 5, 7])["reference_sha256"]
    payload["references"] = {rule["classification_id"]: [0, 1, 2, 3, 4, 5, 6]}
    assert run(**payload)["resolved_rules"]["poverty"]["resolved_cuts"] == fitted["resolved_cuts"]
    rule["none_condition"] = None
    assert classify(-100, fit_rule(rule, [1, 2, 3, 4, 5, 6])) == .33
    with pytest.raises(ValueError, match="collapses"):
        fit_rule(rule, [1, 1, 1, 1])
    with pytest.raises(ValueError, match="reference sample"):
        fit_rule(rule)


def test_contract_excludes_quintiles_and_automatic_critic():
    policy, _ = contract()
    assert "quintiles" not in policy["methods"]
    assert policy["critic"].startswith("diagnostic_only")
    assert policy["class_scores"] == {"none": 0, "low": .33, "medium": .66, "high": 1}


def test_critic_reproduces_published_example():
    # Diakoulaki et al. 1995, tables 2 and 3, DOI 10.1016/0305-0548(94)00059-H.
    cols = {"profitability": [1, .578, .524, .455, .418, .404, .298, 0],
            "market_share": [.212, 0, .446, .785, .650, 1, .845, .790],
            "productivity": [.996, 1, .460, .200, .401, .069, .335, 0]}
    assert critic_weights(cols)["weights"] == pytest.approx(
        {"profitability": .202, "market_share": .481, "productivity": .317}, abs=.001)


def test_critic_constants_and_no_conflict_have_no_fallback():
    result = critic_weights({"a": [0, .5, 1], "b": [0, .5, 1], "c": [4, 4, 4]})
    assert result["weights"] is None
    assert result["constant_columns"] == ["c"]
    result = critic_weights({"a": [0, .5, 1], "b": [1, .5, 0], "c": [4, 4, 4]})
    assert result["weights"] == {"a": .5, "b": .5, "c": 0}


def test_critic_does_not_resolve_policy_ownership_of_duplicates():
    a = [0, 0, 1, 1]
    b = [0, 1, 0, 1]
    original = critic_weights({"a": a, "b": b})
    duplicate = critic_weights({"a": a, "a_copy": a, "b": b})
    assert original["weights"] == {"a": .5, "b": .5}
    assert duplicate["weights"] == {"a": .25, "a_copy": .25, "b": .5}
    # CRITIC divides the duplicate block here, but cannot identify which record is canonical.
    assert duplicate["pearson"]["a"]["a_copy"] == 1


def test_critic_area_and_discretization_can_change_weights():
    cols = {"a": [0, 1, 2, 3, 4, 5], "b": [0, 1, 2, 3, 4, 50],
            "c": [1, 0, 1, 0, 1, 0]}
    full = critic_weights(cols)
    subset = critic_weights({k: v[:4] for k, v in cols.items()})
    assert full["weights"] != subset["weights"]
    quantized = {k: [0 if x == 0 else .33 if x <= 2 else .66 if x <= 4 else 1 for x in v]
                 for k, v in cols.items()}
    assert critic_weights(quantized, scale="fixed_0_1")["weights"] != full["weights"]


def test_alias_selection_preserves_score(payload):
    original = run(**payload)
    payload["config"]["variables"].append(dict(payload["config"]["variables"][1], table="T5"))
    payload["layers"]["T5:flood_example"] = payload["layers"].pop("T2:flood_example")
    payload["config"]["selection"]["flood"] = {"variable": "T5:flood_example",
                                                "reason": "Canonical equivalent in T5"}
    assert run(**payload)["pixels"] == original["pixels"]


def test_reverse_terciles_and_equal_values(payload):
    rule = payload["config"]["rules"][0]
    del rule["cuts"]
    rule.update(method="terciles", direction="lower_is_more_concern",
                reference={"population": "synthetic", "period": "baseline"},
                none_condition={"op": "ge", "value": 7})
    fitted = fit_rule(rule, [1, 2, 3, 4, 5, 6, 7, 7])
    assert [classify(v, fitted) for v in [1, 3, 3, 5, 7, 8]] == [1, .66, .66, .33, 0, 0]
    rule["none_condition"]["op"] = "le"
    with pytest.raises(ValueError, match="contradicts"):
        fit_rule(rule, [1, 2, 3, 4, 5, 6])


def test_cli_roundtrip_and_bad_input_do_not_write(payload, tmp_path):
    import subprocess
    import sys
    source = tmp_path / "inputs.json"
    target = tmp_path / "result.json"
    source.write_text(json.dumps(payload))
    subprocess.run([sys.executable, "-m", "nbs_runtime.priority", str(source),
                    "--output", str(target)], check=True)
    assert json.loads(target.read_text()) == run(**payload)
    target.unlink()
    payload["config"]["weights"]["poverty"] = -1
    source.write_text(json.dumps(payload))
    proc = subprocess.run([sys.executable, "-m", "nbs_runtime.priority", str(source),
                           "--output", str(target)], check=False, capture_output=True)
    assert proc.returncode != 0
    assert not target.exists()


def test_weight_sensitivity_reverses_order(payload):
    before = run(**payload)["pixels"]
    payload["config"]["weights"] = {"poverty": 1, "flood": 3}
    after = run(**payload)["pixels"]
    assert before[0]["score"] > before[1]["score"]
    assert after[0]["score"] < after[1]["score"]


def test_bad_mask_is_not_silent_truthiness(payload):
    payload["opportunity"][0] = 1
    with pytest.raises(ValueError, match="bool or null"):
        run(**payload)


def test_missing_and_constant_critic():
    result = critic_weights({"a": [0, None, 1], "b": [1, 1, 0]})
    assert result["n"] == 2
    assert result["excluded_missing"] == 1
    assert result["weights"] == {"a": .5, "b": .5}
    assert critic_weights({"a": [2, 2], "b": [3, 3]})["weights"] is None
    assert critic_weights({"a": [None], "b": [1]})["weights"] is None
