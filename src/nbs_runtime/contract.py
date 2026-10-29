"""Load the pinned CIAT T4 developer package. Read-only: nothing here changes analytical values.

Entry points are developer_contract.json (rows, function options) and run_contract.json
(approved policies). The raw T4_suitability_mappings.csv is never used.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

UPSTREAM = Path(__file__).resolve().parents[2] / "upstream"


@dataclass(frozen=True)
class Criterion:
    rule_id: str
    function_id: str
    fingerprint: str
    weight: float
    group: str


@dataclass(frozen=True)
class Policy:
    policy_id: str
    aggregation: str
    missing_values: str
    applicability: str
    decision_reference: str
    criteria: tuple[Criterion, ...]
    gate_rule_ids: tuple[str, ...]
    holds: tuple[str, ...]
    sensitivity_aggregation: str | None = None


@dataclass
class Package:
    root: Path
    pin: dict[str, Any]
    contract: dict[str, Any]
    run_contract: dict[str, Any]
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def version(self) -> str:
        return self.pin["package_version"]

    def policy(self, policy_id: str) -> Policy:
        for p in self.run_contract["approved_policies"]:
            if p["policy_id"] == policy_id:
                return Policy(
                    policy_id=p["policy_id"],
                    aggregation=p["aggregation"],
                    missing_values=p["missing_values"],
                    applicability=p["applicability"],
                    decision_reference=p["decision_reference"],
                    criteria=tuple(
                        Criterion(
                            c["rule_id"],
                            c["function_id"],
                            c["function_fingerprint"],
                            float(c["weight"]),
                            c["variable_group"],
                        )
                        for c in p["criteria"]
                    ),
                    gate_rule_ids=tuple(g["rule_id"] for g in p["gate_requirements"]),
                    holds=tuple(p.get("holds", [])),
                    sensitivity_aggregation=p.get("sensitivity_aggregation"),
                )
        raise KeyError(f"policy {policy_id!r} not in package {self.version}")

    def option(self, rule_id: str, function_id: str) -> dict[str, Any]:
        for o in self.rows[rule_id]["function_options"]:
            if o["function_id"] == function_id:
                return o
        raise KeyError(f"{function_id!r} is not an option of {rule_id!r}")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_pin(upstream: Path = UPSTREAM) -> list[str]:
    """Return a list of problems; empty means the vendored files match PIN.json."""
    pin_path = upstream / "PIN.json"
    if not pin_path.exists():
        return ["upstream/PIN.json is missing; run scripts/sync_upstream.py"]
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    problems = []
    for name, expected in pin["files"].items():
        p = upstream / pin["package"] / name
        if not p.exists():
            problems.append(f"{name}: missing")
        elif _sha(p) != expected:
            problems.append(f"{name}: hash differs from PIN.json")
    return problems


def load_package(upstream: Path = UPSTREAM) -> Package:
    problems = verify_pin(upstream)
    if problems:
        raise RuntimeError("Pinned package is not intact: " + "; ".join(problems))
    pin = json.loads((upstream / "PIN.json").read_text(encoding="utf-8"))
    root = upstream / pin["package"]
    contract = json.loads((root / "developer_contract.json").read_text(encoding="utf-8"))
    run_contract = json.loads((root / "run_contract.json").read_text(encoding="utf-8"))
    rows = {r["rule_id"]: r for r in contract["rows"]}
    return Package(root, pin, contract, run_contract, rows)


def fingerprint_holds(pkg: Package, policy: Policy) -> list[str]:
    """A changed or missing function fingerprint holds the policy (CIAT rule)."""
    holds = []
    for c in policy.criteria:
        try:
            opt = pkg.option(c.rule_id, c.function_id)
        except KeyError:
            holds.append(f"{c.rule_id}: function {c.function_id} is missing from the package")
            continue
        if opt.get("fingerprint") != c.fingerprint:
            holds.append(f"{c.rule_id}: fingerprint of {c.function_id} differs from the policy")
    return holds


def weight_holds(policy: Policy, tol: float = 1e-9) -> list[str]:
    total = sum(c.weight for c in policy.criteria)
    out = []
    if any(c.weight < 0 for c in policy.criteria):
        out.append("negative weight in policy")
    if abs(total - 1.0) > tol:
        out.append(f"policy weights sum to {total}, not 1")
    return out
