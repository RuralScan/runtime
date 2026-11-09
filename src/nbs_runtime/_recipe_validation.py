"""Read-only compatibility checks for recipe release v1 (frozen schema 0.4.0)."""

from __future__ import annotations

import csv
import io
import json
import math
import re
from collections import Counter, defaultdict

TABLE_STEMS = {
    "T3": "T3_nbs_hazard_farming",
    "T4": "T4_suitability_mappings",
    "T6": "T6_nbs_scorecard",
    "T6E": "T6E_economic_profile",
}
TABLE_FIELDS = {
    "T3": [
        "record_id",
        "nbs_id",
        "hazard_type",
        "farming_system",
        "mitigation_potential",
        "mitigation_mechanism",
        "confidence",
        "timescale_of_effect",
        "landscape_scale_only",
        "caveats",
        "justification",
        "risk_role",
        "evidence_ids",
        "suitability_family_id",
        "scope_type",
        "scope_id",
        "applicability",
        "family_spread",
        "evidence_level",
        "agreement_level",
        "context_dependent",
        "asset_sensitivity",
        "asset_risk_weight",
    ],
    "T4": [
        "mapping_id",
        "nbs_id",
        "dataset_id",
        "suitability_dimension",
        "relationship_type",
        "relationship_params",
        "uncertainty_pct",
        "context_overrides",
        "is_scenario_candidate",
        "scenario_label",
        "scenario_description",
        "has_future_projection",
        "baseline_dataset_id",
        "future_dataset_ids",
        "weight_default",
        "weight_adjustable",
        "justification",
        "variable",
        "suitability_family_id",
        "evidence_ids",
        "n_sources",
        "corpus_n",
        "paper_support_pct",
    ],
    "T6": [
        "record_id",
        "nbs_id",
        "variable_id",
        "variable_type",
        "effect_direction",
        "effect_confidence",
        "effect_mechanism",
        "conditionality",
        "timescale_of_effect",
        "economic_indicator_type",
        "economic_value_range",
        "justification",
        "economic_archetype_id",
        "evidence_ids",
        "suitability_family_id",
        "scope_type",
        "scope_id",
        "applicability",
        "family_spread",
        "magnitude_summary",
        "evidence_level",
        "agreement_level",
        "context_dependent",
    ],
}
TABLE_FIELDS["T6E"] = TABLE_FIELDS["T6"]
REPORT_FILES = ("T3T6_synthesis_report.json", "T4_readiness.json")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_AUTHORED = {"authored_mappings.csv", "mapping_settings.csv", "review_holds.csv"}
_FLAGS = {"effect_prose": "T3T6_prose.json", "run_policy": "T4_run_policy.csv"}
_POLICY = {
    "aggregation": "weighted_linear_combination",
    "sensitivity_aggregation": "minimum",
    "missing_values": "propagate_unknown",
    "gate_policy": "separate_host_and_legal_masks",
    "applicability": "target_transfer_required",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _name(value):
    return isinstance(value, str) and bool(value.strip())


def _names(values, label):
    _require(isinstance(values, list) and all(_name(v) for v in values), f"Invalid list: {label}")
    _require(len(set(values)) == len(values), f"Duplicate entry: {label}")
    return set(values)


def _decode(text):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            _require(key not in obj, f"Duplicate JSON key: {key}")
            obj[key] = value
        return obj

    def finite(token):
        value = float(token)
        _require(math.isfinite(value), "Non-finite JSON number")
        return value

    return json.loads(text, object_pairs_hook=unique, parse_constant=finite, parse_float=finite)


def _cell(value):
    # The producer decodes JSON scalars/objects/arrays and omits null or empty cells.
    try:
        return _decode(value) if value else None
    except json.JSONDecodeError:
        return value


def _same(left, right):
    # Python equality otherwise treats booleans as integers, including within nested records.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


class _Documents:
    """Track exactly which members the declared release consumes."""

    def __init__(self, members):
        self.members = members
        self.used = set()

    def text(self, path):
        raw = self.members.get(path)
        _require(isinstance(raw, bytes), f"Missing recipe member: {path}")
        self.used.add(path)
        return raw.decode("utf-8")

    def json(self, path, kind=dict):
        value = _decode(self.text(path))
        _require(isinstance(value, kind), f"Invalid JSON structure: {path}")
        return value

    def csv(self, path, columns, *, frozen=False):
        reader = csv.DictReader(io.StringIO(self.text(path), newline=""), strict=True)
        try:
            header = reader.fieldnames or []
            _require(
                all(_name(f) for f in header) and len(header) == len(set(header)),
                f"Invalid CSV header: {path}",
            )
            _require(
                header == columns if frozen else set(columns) <= set(header),
                f"Unexpected CSV columns: {path}",
            )
            rows = list(reader)
        except csv.Error as exc:
            raise ValueError(f"Invalid CSV: {path}") from exc
        _require(
            all(None not in r and None not in r.values() for r in rows),
            f"Invalid CSV row width: {path}",
        )
        return rows


def _tables(docs, practice, seen):
    tables = {}
    for table, stem in TABLE_STEMS.items():
        path = f"tables/{practice}/{stem}"
        key = "mapping_id" if table == "T4" else "record_id"
        rows = {}
        for raw in docs.csv(path + ".csv", TABLE_FIELDS[table], frozen=True):
            row = {name: val for name, cell in raw.items() if (val := _cell(cell)) is not None}
            rid = row.get(key)
            _require(
                _name(rid) and rid not in seen[table] and row.get("nbs_id") == practice,
                f"Invalid, duplicate or misplaced row in {path}: {rid}",
            )
            if "evidence_ids" in row:
                _names(row["evidence_ids"], f"evidence IDs for {rid}")
            seen[table].add(rid)
            rows[rid] = row
        _require(
            _same(list(rows.values()), docs.json(path + ".json", list)),
            f"CSV and typed JSON differ: {path}",
        )
        tables[table] = rows
    return tables


def _settings(docs, practice, declaration):
    base = f"settings/{practice}/"
    _require(
        isinstance(declaration, dict) and set(declaration) == {"settings"},
        f"Invalid settings declaration for {practice}",
    )
    index = docs.json(base + "T4_settings.json")
    _require(index.get("nbs_id") == practice, f"Wrong practice in settings for {practice}")
    leaves = _names(index.get("files"), "T4 settings")
    _require(leaves <= _AUTHORED, f"Unsupported authored setting for {practice}")
    for flag, filename in _FLAGS.items():
        _require(type(index.get(flag)) is bool, f"Invalid settings flag: {flag}")
        if index[flag]:
            leaves.add(filename)
    required = leaves | {"T0_nbs_registry.csv", "T4_settings.json"}
    _require(
        _names(declaration["settings"], practice) == {base + name for name in required},
        f"Incomplete settings inventory for {practice}",
    )
    identity = docs.csv(base + "T0_nbs_registry.csv", ["nbs_id"])
    _require(
        len(identity) == 1 and identity[0]["nbs_id"] == practice,
        f"Wrong or missing T0 identity for {practice}",
    )
    configs = {}
    for name in leaves:
        if name.endswith(".json"):
            docs.json(base + name)
        else:
            rows = docs.csv(base + name, ["nbs_id"])
            _require(
                rows and all(r["nbs_id"] == practice for r in rows),
                f"Empty or misplaced authored setting: {base}{name}",
            )
            configs[name] = rows
    return configs


def _adaptation(carrier):
    _require(isinstance(carrier, dict), "Invalid T4 relationship parameters or profile")
    adaptation = carrier.get("reviewed_adaptation", {})
    _require(isinstance(adaptation, dict), "Invalid reviewed T4 adaptation")
    for name, status in (
        ("numerical_calibration", "held"),
        ("dataset_binding", "requires_matching_input"),
    ):
        if name in adaptation:
            item = adaptation[name]
            _require(
                isinstance(item, dict)
                and item.get("status") == status
                and _name(item.get("reason")),
                f"Invalid {name} hold",
            )
    return adaptation


def _holds(rows, settings):
    declared = {}
    for raw in settings:
        row = {key: _cell(cell) for key, cell in raw.items()}
        key = row.get("mapping_id")
        adaptation = _adaptation(row.get("relationship_params") or {})
        _require(
            _name(key)
            and key not in declared
            and key in rows
            and ("numerical_calibration" in adaptation or "dataset_binding" in adaptation),
            f"Invalid authored T4 hold: {key}",
        )
        _require(
            row.get("variable") == rows[key].get("variable")
            and row.get("suitability_family_id") == rows[key].get("suitability_family_id"),
            f"Authored hold changed scope: {key}",
        )
        if "numerical_calibration" in adaptation:
            _require(
                all(
                    _same(row.get(k), rows[key].get(k))
                    for k in ("relationship_type", "relationship_params", "evidence_ids")
                ),
                f"Calibration hold relationship or evidence differs: {key}",
            )
        declared[key] = adaptation
    held = set()
    for key, row in rows.items():
        params = row.get("relationship_params", {})
        adaptation = _adaptation(params)
        if (
            key in declared
            or "numerical_calibration" in adaptation
            or "dataset_binding" in adaptation
        ):
            _require(
                _same(declared.get(key), adaptation),
                f"Authored hold differs from compiled T4: {key}",
            )
        profiles = params.get("profiles", [])
        _require(isinstance(profiles, list), f"Invalid T4 profile list: {key}")
        profile_holds = [_adaptation(profile) for profile in profiles]
        if "dataset_binding" in adaptation or any("dataset_binding" in p for p in profile_holds):
            _require(
                not row.get("dataset_id"), f"Matching-input hold has a dataset substitution: {key}"
            )
        if "numerical_calibration" in adaptation:
            held.add(key)
    return held


def _policies(records, report, rules):
    declared = _names(report.get("approved_policy_ids"), "approved policies")
    _require("approved_policies" not in report, "Unexpected inline run policies")
    groups = defaultdict(list)
    row_ids = set()
    for row in records:
        rid = row.get("policy_row_id")
        _require(
            _name(rid) and rid not in row_ids and _name(row.get("policy_id")),
            "Absent or duplicate run policy identity",
        )
        row_ids.add(rid)
        groups[row["policy_id"]].append(row)
    _require(declared == groups.keys(), "Authored policies differ from approved policy list")
    authorized_functions = set()
    for pid, members in groups.items():
        first = members[0]
        system = first.get("system")
        _require(
            _name(system)
            and _name(first.get("decision_reference"))
            and _name(first.get("output_label")),
            f"Incomplete run policy: {pid}",
        )
        roles = {r["role"] for r in rules.values() if r["system"] == system}
        _require(
            {"eligibility_gate", "hard_exclusion"} <= roles, f"Missing required policy gates: {pid}"
        )
        budgets, totals, criteria = {}, defaultdict(float), set()
        for row in members:
            _require(
                all(row.get(k) == v for k, v in _POLICY.items())
                and all(
                    row.get(k) == first.get(k)
                    for k in ("system", "decision_reference", "output_label")
                ),
                f"Unsupported or inconsistent policy metadata: {pid}",
            )
            key = row.get("rule_id")
            rule = rules.get(key)
            group = row.get("variable_group")
            _require(
                rule
                and key not in criteria
                and rule["role"] == "suitability_score"
                and rule["system"] == system
                and rule["variable_group"] == group,
                f"Unknown, duplicate or misplaced criterion for {pid}: {key}",
            )
            criteria.add(key)
            function, fingerprint = row.get("function_id"), row.get("function_fingerprint")
            _require(
                _name(fingerprint) and rule["functions"].get(function) == fingerprint,
                f"Policy function fingerprint disagrees with readiness for {key}",
            )
            authorized_functions.add((key, function))
            try:
                weight = float(row["criterion_weight"])
                budget = float(row["group_weight"])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Invalid policy weight: {key}") from exc
            _require(
                0 < weight <= 1 and 0 < budget <= 1 and budget == budgets.setdefault(group, budget),
                f"Inconsistent criterion weight or budget for {key}",
            )
            totals[group] += weight
        _require(
            math.isclose(sum(budgets.values()), 1, abs_tol=1e-9)
            and all(math.isclose(totals[g], b, abs_tol=1e-9) for g, b in budgets.items()),
            f"Policy weights do not match group budgets: {pid}",
        )
    for key, rule in rules.items():
        status = rule["readiness"]
        if status.get("function_basis") == "approved_policy":
            _require(
                (key, status.get("effective_function_id")) in authorized_functions,
                f"Effective function has no matching authored policy: {key}",
            )


def _reports(docs, practice, tables, configs):
    base = f"tables/{practice}/"
    effects = docs.json(base + REPORT_FILES[0])
    _require(effects.get("nbs_id") == practice, f"Wrong practice in effect report: {practice}")
    for table in ("T3", "T6", "T6E"):
        count = effects.get(table.lower() + "_rows")
        _require(
            type(count) is int and count == len(tables[table]),
            f"Wrong {table} row count for {practice}",
        )
    if "units_by_role" in effects or "n_units_pooled" in effects:
        roles = effects.get("units_by_role")
        total = effects.get("n_units_pooled")
        _require(
            isinstance(roles, dict)
            and all(type(n) is int and n >= 0 for n in roles.values())
            and type(total) is int
            and sum(roles.values()) == total,
            f"Inconsistent pooled evidence counts for {practice}",
        )

    report = docs.json(base + REPORT_FILES[1])
    _require(
        report.get("nbs_id") == practice
        and report.get("contract_version") == "0.2.0"
        and report.get("execution_ready") is False
        and isinstance(report.get("rows"), list),
        f"Unsupported T4 readiness report for {practice}",
    )
    held = _holds(tables["T4"], configs.get("review_holds.csv", []))
    rules, states = {}, Counter()
    for rule in report["rows"]:
        _require(
            isinstance(rule, dict) and isinstance(rule.get("readiness"), dict),
            f"Invalid readiness record for {practice}",
        )
        key, role, status = rule.get("rule_id"), rule.get("role"), rule["readiness"]
        _require(
            _name(key)
            and key not in rules
            and _name(rule.get("system"))
            and _name(rule.get("variable_group")),
            f"Invalid readiness identity for {practice}",
        )
        state = status.get("state")
        _require(
            role
            in (
                "suitability_score",
                "hard_exclusion",
                "eligibility_gate",
                "operational_scenario",
                "reference_only",
            )
            and state in ("defined", "provisional", "blocked", "reference_only")
            and (state == "reference_only") == (role == "reference_only")
            and (key in tables["T4"]) == (state != "reference_only")
            and status.get("execution_ready") is False,
            f"Invalid readiness role or state: {key}",
        )
        options = rule.get("functions")
        _require(
            isinstance(options, dict)
            and all(
                _name(k) and isinstance(v, str) and _DIGEST.fullmatch(v) for k, v in options.items()
            ),
            f"Invalid function list: {key}",
        )
        if state == "reference_only":
            _require(not options, f"Reference rule includes functions: {key}")
        selected = status.get("effective_function_id")
        _require(
            (selected is None or (_name(selected) and selected in options))
            and status.get("effective_function_fingerprint") == options.get(selected),
            f"Selected function fingerprint disagrees: {key}",
        )
        if key in tables["T4"] and tables["T4"][key].get("suitability_family_id"):
            _require(
                tables["T4"][key]["suitability_family_id"] == rule["system"],
                f"T4 readiness has a different system: {key}",
            )
        if key in held:
            _require(
                state == "blocked" and not options and selected is None,
                f"Calibration hold is not blocked: {key}",
            )
        rules[key] = rule
        states[state] += 1
    _require(
        tables["T4"].keys() <= rules.keys()
        and _same(states, report.get("implementation_readiness")),
        f"Readiness coverage disagrees with T4 for {practice}",
    )
    _policies(configs.get("T4_run_policy.csv", []), report, rules)


def validate_recipe(members: dict[str, bytes]) -> tuple[str, ...]:
    """Validate every declared practice, all four tables, settings and review reports."""
    docs = _Documents(members)
    boot = docs.json("recipe-release.json")
    _require(
        set(boot) == {"format_version", "kind", "practices"}
        and type(boot["format_version"]) is int
        and boot["format_version"] == 1
        and boot["kind"] == "reconciled_recipe",
        "Unsupported recipe-release format or kind",
    )
    practices = boot["practices"]
    _require(isinstance(practices, dict) and practices, "Release must contain a practice inventory")
    provenance = docs.json("provenance.json")
    _require(
        provenance.get("schema_version") == "0.4.0-structure-frozen"
        and _name(provenance.get("source_revision")),
        "Unsupported recipe schema or missing source revision",
    )
    for name in ("inputs_sha256", "compiler_sha256"):
        hashes = provenance.get(name)
        _require(
            isinstance(hashes, dict)
            and hashes
            and all(
                _name(k) and isinstance(v, str) and _DIGEST.fullmatch(v) for k, v in hashes.items()
            ),
            f"Invalid {name} provenance",
        )
    seen = {table: set() for table in TABLE_STEMS}
    for practice, declaration in practices.items():
        _require(re.fullmatch(r"[a-z][a-z0-9_]*", practice), f"Invalid practice ID: {practice}")
        configs = _settings(docs, practice, declaration)
        tables = _tables(docs, practice, seen)
        _reports(docs, practice, tables, configs)
    _require(members.keys() == docs.used, "Release contains undeclared members")
    return tuple(sorted(practices))
