"""Apply explicit T2/T5 classifications and user weights to aligned pixels.

Scientific rules belong to the pinned method contract and supplied configuration.
This evaluator does not calibrate thresholds, choose representatives or approve science.
"""
from __future__ import annotations

import argparse
import json
import re
from bisect import bisect_left
from collections import Counter
from functools import lru_cache
from hashlib import sha256
from importlib.resources import files
from itertools import pairwise
from math import isfinite
from pathlib import Path
from statistics import mean, pstdev


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=1)
def contract():
    raw = files("nbs_runtime").joinpath("priority_contract.json").read_bytes()
    pin = json.loads(files("nbs_runtime").joinpath("priority_pin.json").read_bytes())
    require(pin["repository"] == "RuralScan/method", "Unknown priority contract publisher")
    require(re.fullmatch(r"[0-9a-f]{40}", pin["revision"]), "Invalid method revision")
    require(sha256(raw).hexdigest() == pin["sha256"], "Priority contract checksum mismatch")
    return json.loads(raw), pin


def number(value):
    require(isinstance(value, (int, float)) and not isinstance(value, bool), "Expected a number")
    require(isfinite(value), "Expected a finite number")
    return float(value)


def observations(values):
    require(isinstance(values, list), "Values must be an aligned list")
    return [None if v is None or (isinstance(v, float) and not isfinite(v)) else number(v)
            for v in values]


def fields(obj, required, optional=()):
    require(isinstance(obj, dict), "Expected an object")
    require(set(required) <= obj.keys(), f"Missing fields: {set(required) - obj.keys()}")
    require(obj.keys() <= set(required) | set(optional), "Unknown configuration fields")


def text_fields(obj, names):
    for key in names:
        require(isinstance(obj[key], str) and bool(obj[key].strip()), f"Missing text: {key}")


def no_problem(value, condition):
    if condition is None:
        return False
    return value <= condition["value"] if condition["op"] == "le" else value >= condition["value"]


def fit_rule(rule, reference=None):
    """Resolve a declared rule; reference samples are never inferred from target pixels."""
    policy, _ = contract()
    base = ("classification_id", "version", "method", "direction", "unit", "context",
            "decision_ref", "missing")
    fields(rule, base, ("cuts", "reference", "none_condition"))
    text_fields(rule, base)
    require(rule["method"] in policy["methods"], "Unsupported classification method")
    require(rule["direction"] in policy["directions"], "Unsupported direction")
    require(rule["missing"] == "unknown", "Missing values must remain unknown")
    fitted = dict(rule)
    if rule["method"] == "thresholds":
        require(set(rule) == set(base) | {"cuts"}, "Thresholds require only explicit cuts")
        require(isinstance(rule["cuts"], list), "Cutoffs must be a list")
        cuts = [number(x) for x in rule["cuts"]]
        require(len(cuts) == len(policy["class_order"]) - 1, "Expected three cutoffs")
        require(all(a < b for a, b in pairwise(cuts)), "Cutoffs must strictly increase")
        fitted["resolved_cuts"] = cuts
    else:
        require(set(rule) == set(base) | {"reference", "none_condition"},
                "Terciles require a reference and explicit none_condition (possibly null)")
        fields(rule["reference"], ("population", "period"))
        text_fields(rule["reference"], ("population", "period"))
        condition = rule["none_condition"]
        if condition is not None:
            fields(condition, ("op", "value"))
            expected = "le" if rule["direction"] == "higher_is_more_concern" else "ge"
            require(condition["op"] == expected, "No-problem condition contradicts direction")
            number(condition["value"])
        require(reference is not None, "Explicit reference sample required")
        sample = observations(reference)
        retained = sorted(x for x in sample if x is not None and not no_problem(x, condition))
        require(len(retained) >= 3, "Terciles require at least three usable reference values")
        cuts = []
        for q in policy["tercile_quantiles"]:
            index = (len(retained) - 1) * q
            lo = int(index)
            cuts.append(retained[lo] + (retained[min(lo + 1, len(retained) - 1)]
                                       - retained[lo]) * (index - lo))
        require(cuts[0] < cuts[1], "Tied reference sample collapses tercile cutoffs")
        fitted.update(resolved_cuts=cuts, reference_sha256=digest(sample),
                      reference_count=len(sample), fitted_count=len(retained))
    return fitted


def classify(value, fitted):
    policy, _ = contract()
    if value is None:
        return None
    value = number(value)
    labels = policy["class_order"]
    if fitted["method"] == "terciles":
        if no_problem(value, fitted["none_condition"]):
            return policy["class_scores"]["none"]
        labels = labels[1:]
    if fitted["direction"] == "lower_is_more_concern":
        labels = labels[::-1]
    return policy["class_scores"][labels[bisect_left(fitted["resolved_cuts"], value)]]


def correlation(columns):
    """Pearson correlations on one shared complete-case sample; constants stay undefined."""
    require(bool(columns), "No columns")
    cols = {k: observations(v) for k, v in columns.items()}
    require(len({len(v) for v in cols.values()}) == 1, "Unaligned columns")
    names = list(cols)
    rows = [row for row in zip(*cols.values()) if all(v is not None for v in row)]
    complete = {name: [row[j] for row in rows] for j, name in enumerate(names)}
    sd = {name: pstdev(vals) if vals else 0.0 for name, vals in complete.items()}
    matrix = {}
    for a in names:
        matrix[a] = {}
        for b in names:
            r = None
            if len(rows) >= 2 and sd[a] > 0 and sd[b] > 0:
                av, bv = mean(complete[a]), mean(complete[b])
                r = mean((x - av) * (y - bv) for x, y in zip(complete[a], complete[b]))
                r = max(-1.0, min(1.0, r / (sd[a] * sd[b])))
                if abs(abs(r) - 1) < 1e-12:
                    r = 1.0 if r > 0 else -1.0
            matrix[a][b] = r
    return {"n": len(rows), "excluded_missing": len(next(iter(cols.values()))) - len(rows),
            "sd": sd, "pearson": matrix}


def critic_weights(columns, *, scale="minmax"):
    """Diagnostic only: original min-max CRITIC or explicitly fixed 0–1 inputs.

    Diakoulaki et al. (1995), DOI: 10.1016/0305-0548(94)00059-H, equations 2–6.
    Constant criteria are reported and omitted from the conflict calculation.
    """
    require(scale in {"minmax", "fixed_0_1"}, "Unknown CRITIC scaling")
    require(bool(columns), "No columns")
    clean = {k: observations(v) for k, v in columns.items()}
    require(len({len(v) for v in clean.values()}) == 1, "Unaligned columns")
    rows = [row for row in zip(*clean.values()) if all(v is not None for v in row)]
    normalized = {}
    constants = []
    for j, name in enumerate(clean):
        vals = [row[j] for row in rows]
        if scale == "fixed_0_1":
            require(all(0 <= v <= 1 for v in vals), "Fixed-scale scores must lie in 0–1")
        if not vals or max(vals) == min(vals):
            constants.append(name)
            continue
        low, high = min(vals), max(vals)
        normalized[name] = ([(v - low) / (high - low) for v in vals]
                            if scale == "minmax" else vals)
    diag = correlation(normalized) if normalized else {"n": len(rows), "pearson": {}, "sd": {}}
    information = {k: diag["sd"][k] * sum(1 - r for r in diag["pearson"][k].values())
                   for k in normalized}
    total = sum(information.values())
    weights = ({k: information.get(k, 0.0) / total for k in clean} if total > 1e-12 else None)
    return {"diagnostic_only": True, "scale": scale, "n": len(rows),
            "excluded_missing": len(next(iter(clean.values()))) - len(rows),
            "constant_columns": constants, "weights": weights,
            "reason": None if weights is not None else "No identifiable CRITIC weights",
            "information": information, "pearson": diag["pearson"]}


def run(config, layers, opportunity, references=None):
    """Return classified layers, correlation diagnostics, weights and per-pixel contributions."""
    policy, pin = contract()
    fields(config, ("format", "purpose", "context", "variables", "rules", "selection", "weights"))
    require(config["format"] == policy["format"], "Unsupported priority configuration")
    require(config["purpose"] in {"illustration", "analysis"}, "Unknown run purpose")
    text_fields(config, ("context",))
    require(isinstance(opportunity, list) and all(x is None or type(x) is bool for x in opportunity),
            "Opportunity mask must contain bool or null values")
    require(isinstance(config["variables"], list) and isinstance(config["rules"], list),
            "Variables and rules must be lists")
    require(isinstance(config["selection"], dict) and isinstance(config["weights"], dict),
            "Selection and weights must be objects")
    variables, rules = {}, {}
    for var in config["variables"]:
        names = ("table", "variable_id", "unit", "mcda_role", "concern_id", "classification_id")
        fields(var, names, ("variable", "dataset_id", "hazard_type", "scenario_type", "theme"))
        text_fields(var, names[:-1])
        require(var["table"] in policy["table_fields"], "Unknown analytical table")
        require(var["mcda_role"] in policy["roles"], "Unknown variable role")
        require(var["table"] != "T5" or var["mcda_role"] != "investment_risk",
                "T5 cannot be an investment-risk screen")
        key = f"{var['table']}:{var['variable_id']}"
        require(key not in variables, "Duplicate variable identity")
        variables[key] = var
    for rule in config["rules"]:
        require(isinstance(rule, dict) and isinstance(rule.get("classification_id"), str),
                "Missing rule identity")
        key = rule["classification_id"]
        require(key not in rules, "Duplicate classification rule")
        rules[key] = rule
    require(config["weights"].keys() == config["selection"].keys(),
            "Weights and selected concerns must match exactly")
    raw_weights = {k: number(v) for k, v in config["weights"].items()}
    require(all(v >= 0 for v in raw_weights.values()), "Negative weight")
    require(sum(raw_weights.values()) > 0, "All weights are zero")
    scaled = {k: v / max(raw_weights.values()) for k, v in raw_weights.items() if v > 0}
    weights = {k: v / sum(scaled.values()) for k, v in scaled.items()}
    classified, resolved, input_pins, raw = {}, {}, {}, {}
    for concern, choice in config["selection"].items():
        fields(choice, ("variable", "reason"))
        text_fields(choice, ("variable", "reason"))
        var = variables.get(choice["variable"])
        require(var is not None and var["concern_id"] == concern, "Invalid concern representative")
        require(var["mcda_role"] == "priority", "Only priority variables may enter the stack")
        if concern not in weights:
            continue
        rule = rules.get(var["classification_id"])
        require(rule is not None, "Missing classification; no historical fallback")
        require(rule.get("unit") == var["unit"], "Rule/variable unit mismatch")
        if config["purpose"] == "analysis":
            require(not str(rule.get("decision_ref", "")).startswith("illustrative:"),
                    "Illustrative rules cannot run as an analysis")
        fitted = fit_rule(rule, (references or {}).get(var["classification_id"]))
        layer = layers.get(choice["variable"])
        fields(layer, ("values", "unit", "source_ref"))
        text_fields(layer, ("unit", "source_ref"))
        require(layer["unit"] == var["unit"], "Input/variable unit mismatch")
        vals = observations(layer["values"])
        require(len(vals) == len(opportunity), "Input grid length mismatch")
        raw[concern] = vals
        classified[concern] = [classify(v, fitted) if mask is True else None
                               for v, mask in zip(vals, opportunity)]
        resolved[concern] = fitted
        input_pins[concern] = {"variable": choice["variable"], "source_ref": layer["source_ref"],
                               "unit": layer["unit"], "values_sha256": digest(vals)}
    pixels = []
    for i, mask in enumerate(opportunity):
        status = "excluded" if mask is False else "unknown"
        contributions, score = {}, None
        if mask is True and all(classified[k][i] is not None for k in weights):
            status = "scored"
            contributions = {k: weights[k] * classified[k][i] for k in weights}
            score = sum(contributions.values())
        pixels.append({"status": status, "score": score, "contributions": contributions})
    in_mask = {k: [v for v, mask in zip(vals, opportunity) if mask is True]
               for k, vals in classified.items()}
    raw_in_mask = {k: [v for v, mask in zip(vals, opportunity) if mask is True]
                   for k, vals in raw.items()}
    return {"format": policy["format"], "purpose": config["purpose"], "context": config["context"],
            "contract_pin": pin, "configuration_sha256": digest(config),
            "opportunity_sha256": digest(opportunity), "inputs": input_pins,
            "resolved_rules": resolved, "selection": config["selection"], "weights": weights,
            "classified": classified, "pixels": pixels,
            "counts": dict(Counter(p["status"] for p in pixels)),
            "correlations": {"raw": correlation(raw_in_mask), "classified": correlation(in_mask)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON with config, layers, opportunity and references")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    result = run(**payload)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
