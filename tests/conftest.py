import json

import pytest

from nbs_runtime.contract import load_package


@pytest.fixture(scope="session")
def pkg():
    return load_package()


@pytest.fixture(scope="session")
def conformance(pkg):
    return json.loads((pkg.root / "conformance_examples.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def policy_cases(pkg):
    return json.loads((pkg.root / "policy_conformance.json").read_text(encoding="utf-8"))


@pytest.fixture
def recipe_members():
    """A complete synthetic release with one policy, one held rule and an empty table."""
    import csv
    import io

    from nbs_runtime._recipe_validation import REPORT_FILES, TABLE_FIELDS, TABLE_STEMS

    practice = "example"
    settings = f"settings/{practice}/"
    tables = f"tables/{practice}/"
    members = {}

    def put(path, value):
        members[path] = json.dumps(value).encode()

    def csv_file(path, columns, rows):
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v if isinstance(v, str) else json.dumps(v) for k, v in row.items()})
        members[path] = output.getvalue().encode()

    hold = {
        "reviewed_adaptation": {
            "numerical_calibration": {"status": "held", "reason": "Await local calibration."}
        }
    }
    t4 = [
        {"mapping_id": k, "nbs_id": practice, "suitability_family_id": "system", "variable": k}
        for k in ("score", "gate", "mask", "held")
    ]
    t4[-1].update(relationship_params=hold, evidence_ids=["source_unit"])
    data = {
        "T3": [{"record_id": "example_risk", "nbs_id": practice, "landscape_scale_only": True}],
        "T4": t4,
        "T6": [{"record_id": "example_effect", "nbs_id": practice, "evidence_ids": ["unit_a"]}],
        "T6E": [],
    }
    for table, rows in data.items():
        stem = tables + TABLE_STEMS[table]
        csv_file(stem + ".csv", TABLE_FIELDS[table], rows)
        put(stem + ".json", rows)
    csv_file(
        settings + "T0_nbs_registry.csv",
        ["nbs_id", "nbs_name"],
        [{"nbs_id": practice, "nbs_name": "Example"}],
    )
    csv_file(settings + "review_holds.csv", TABLE_FIELDS["T4"], [t4[-1]])
    fingerprint = "a" * 64
    policy = {
        "policy_row_id": "criterion",
        "policy_id": "policy",
        "nbs_id": practice,
        "system": "system",
        "output_label": "Index",
        "variable_group": "soil",
        "rule_id": "score",
        "function_id": "f",
        "function_fingerprint": fingerprint,
        "group_weight": "1",
        "criterion_weight": "1",
        "aggregation": "weighted_linear_combination",
        "sensitivity_aggregation": "minimum",
        "missing_values": "propagate_unknown",
        "gate_policy": "separate_host_and_legal_masks",
        "applicability": "target_transfer_required",
        "decision_reference": "synthetic test decision",
    }
    csv_file(settings + "T4_run_policy.csv", list(policy), [policy])
    put(
        settings + "T4_settings.json",
        {
            "nbs_id": practice,
            "files": ["review_holds.csv"],
            "effect_prose": False,
            "run_policy": True,
        },
    )
    roles = {
        "score": "suitability_score",
        "gate": "eligibility_gate",
        "mask": "hard_exclusion",
        "held": "suitability_score",
    }
    rules = [
        {
            "rule_id": key,
            "role": role,
            "system": "system",
            "variable_group": "soil",
            "functions": {"f": fingerprint} if key == "score" else {},
            "readiness": {
                "state": "blocked" if key == "held" else "defined",
                "execution_ready": False,
            },
        }
        for key, role in roles.items()
    ]
    rules[0]["readiness"].update(
        function_basis="approved_policy",
        effective_function_id="f",
        effective_function_fingerprint=fingerprint,
    )
    put(
        tables + REPORT_FILES[0],
        {
            "nbs_id": practice,
            "t3_rows": 1,
            "t6_rows": 1,
            "t6e_rows": 0,
            "units_by_role": {"example_role": 1},
            "n_units_pooled": 1,
        },
    )
    put(
        tables + REPORT_FILES[1],
        {
            "contract_version": "0.2.0",
            "execution_ready": False,
            "nbs_id": practice,
            "approved_policy_ids": ["policy"],
            "rows": rules,
            "implementation_readiness": {"defined": 3, "blocked": 1},
        },
    )
    put(
        "recipe-release.json",
        {
            "format_version": 1,
            "kind": "reconciled_recipe",
            "practices": {
                practice: {"settings": sorted(k for k in members if k.startswith(settings))}
            },
        },
    )
    put(
        "provenance.json",
        {
            "schema_version": "0.4.0-structure-frozen",
            "source_revision": "1" * 40,
            "inputs_sha256": {"example.csv": "b" * 64},
            "compiler_sha256": {"example.py": "c" * 64},
        },
    )
    return members
