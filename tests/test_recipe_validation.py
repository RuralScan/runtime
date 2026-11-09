"""The analytical contract must remain internally consistent even if re-pinned."""

import csv
import io
import json

import pytest

from nbs_runtime._recipe_validation import validate_recipe

T3 = "tables/example/T3_nbs_hazard_farming.json"
T4 = "tables/example/T4_suitability_mappings.json"
READINESS = "tables/example/T4_readiness.json"
EFFECTS = "tables/example/T3T6_synthesis_report.json"


def test_complete_release_with_one_practice_and_empty_table(recipe_members):
    assert validate_recipe(recipe_members) == ("example",)


def test_readiness_cannot_enable_documentary_hold(recipe_members):
    report = json.loads(recipe_members[READINESS])
    rule = next(r for r in report["rows"] if r["rule_id"] == "held")
    rule["readiness"]["state"] = "defined"
    rule["functions"] = {"new": "e" * 64}
    report["implementation_readiness"] = {"defined": 4}
    recipe_members[READINESS] = json.dumps(report).encode()
    with pytest.raises(ValueError, match="Calibration hold"):
        validate_recipe(recipe_members)


def test_wrong_selected_function_with_valid_alternate_fingerprint(recipe_members):
    report = json.loads(recipe_members[READINESS])
    row = next(r for r in report["rows"] if r["rule_id"] == "score")
    row["functions"]["other"] = "d" * 64
    row["readiness"].update(effective_function_id="other", effective_function_fingerprint="d" * 64)
    recipe_members[READINESS] = json.dumps(report).encode()
    with pytest.raises(ValueError, match="no matching authored policy"):
        validate_recipe(recipe_members)


@pytest.mark.parametrize(
    ("member", "old", "new"),
    [
        ("recipe-release.json", b'"format_version": 1', b'"format_version": true'),
        ("recipe-release.json", b'"settings":', b'"settings": [], "settings":'),
        ("provenance.json", b"0.4.0-structure-frozen", b"2.0"),
        ("provenance.json", b"b" * 64, b"invalid-digest"),
        ("settings/example/T0_nbs_registry.csv", b"example,Example", b"other,Example"),
        ("settings/example/T4_settings.json", b'"run_policy": true', b'"run_policy": false'),
        (
            "settings/example/review_holds.csv",
            b"Await local calibration.",
            b"Conflicting decision.",
        ),
        (T3, b"true", b"1"),
        (T3.replace(".json", ".csv"), b"hazard_type", b"other_type"),
        (T4, b'"source_unit"', b'"unrelated"'),
        (READINESS, b'"contract_version": "0.2.0"', b'"contract_version": "0.1.0"'),
        (READINESS, b'"execution_ready": false', b'"execution_ready": true'),
        (READINESS, b'"defined": 3', b'"defined": 1'),
        (READINESS, b'"approved_policy_ids": ["policy"]', b'"approved_policy_ids": []'),
        ("settings/example/T4_run_policy.csv", b"a" * 64, b"f" * 64),
        ("settings/example/T4_run_policy.csv", b",1,1,", b",1,0.5,"),
        (EFFECTS, b'"t6e_rows": 0', b'"t6e_rows": false'),
        (EFFECTS, b'"n_units_pooled": 1', b'"n_units_pooled": 0'),
    ],
)
def test_reject_inconsistent_semantics(recipe_members, member, old, new):
    assert old in recipe_members[member]
    recipe_members[member] = recipe_members[member].replace(old, new)
    with pytest.raises(ValueError):
        validate_recipe(recipe_members)


def test_missing_and_extra_members(recipe_members):
    extra = dict(recipe_members, other=b"unused")
    with pytest.raises(ValueError, match="undeclared"):
        validate_recipe(extra)
    del recipe_members["tables/example/T6E_economic_profile.csv"]
    with pytest.raises(ValueError, match="Missing recipe member"):
        validate_recipe(recipe_members)


def test_reject_held_relationship_replacement(recipe_members):
    rows = json.loads(recipe_members[T4])
    held = next(r for r in rows if r["mapping_id"] == "held")
    held["relationship_params"]["threshold"] = 10
    _rewrite_t4(recipe_members, rows)
    with pytest.raises(ValueError, match="hold relationship or evidence"):
        validate_recipe(recipe_members)


def test_reject_dataset_substitution_on_profile_input_hold(recipe_members):
    rows = json.loads(recipe_members[T4])
    score = next(r for r in rows if r["mapping_id"] == "score")
    score["dataset_id"] = "substitute"
    score["relationship_params"] = {
        "profiles": [
            {
                "reviewed_adaptation": {
                    "dataset_binding": {
                        "status": "requires_matching_input",
                        "reason": "Check units.",
                    }
                }
            }
        ]
    }
    _rewrite_t4(recipe_members, rows)
    with pytest.raises(ValueError, match="Matching-input hold"):
        validate_recipe(recipe_members)


def _rewrite_t4(members, rows):
    """Rebuild both formats so the test reaches the semantic checks."""
    path = T4.replace(".json", ".csv")
    header = next(csv.reader(io.StringIO(members[path].decode())))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, header)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: v if isinstance(v, str) else json.dumps(v) for k, v in row.items()})
    members[path] = stream.getvalue().encode()
    members[T4] = json.dumps(rows).encode()
