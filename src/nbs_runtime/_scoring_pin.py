"""Read and validate the seven files in a scoring-package pin."""

import hashlib
import json
import re
from pathlib import Path

FILES = (
    "developer_contract.json",
    "run_contract.json",
    "conformance_examples.json",
    "policy_conformance.json",
    "selection_template.json",
    "manifest.json",
    "VERSION",
)
CONTRACT_VERSION = "0.2.0"


def relative_path(value: str) -> str:
    """A portable repository-relative path, also safe to use in a GitHub URL."""
    if not isinstance(value, str) or not all(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part) for part in value.split("/")
    ):
        raise ValueError(f"Invalid repository-relative path: {value!r}")
    return value


def publisher_root(repo: str, source_root: str) -> str:
    if len(relative_path(repo).split("/")) != 2:
        raise ValueError("Specify a GitHub publisher as owner/repository")
    return relative_path(source_root)


def package_dir(upstream: Path, package: str) -> Path:
    root = upstream.resolve() / relative_path(package)
    if root.resolve() != root:
        raise ValueError(f"Redirected package directory: {root}")
    return root


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_pinned(upstream: Path) -> tuple[dict, Path, dict, dict]:
    """Use one PIN.json snapshot for integrity checks and decoded runtime inputs."""
    pin = json.loads((upstream / "PIN.json").read_bytes())
    if not isinstance(pin, dict):
        raise TypeError("Invalid scoring-package pin")
    root = package_dir(upstream, pin.get("package"))
    publisher_root(pin.get("source_repo"), pin.get("source_root", "handoff"))
    hashes = pin.get("files")
    if not isinstance(hashes, dict) or set(hashes) != set(FILES):
        raise ValueError("PIN.json must list all seven scoring-package files")
    if not isinstance(pin.get("ref"), str) or not re.fullmatch(r"[0-9a-f]{40}", pin["ref"]):
        raise ValueError("PIN.json requires an immutable Git commit SHA")
    if pin.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(f"Unsupported scoring contract: {pin.get('contract_version')!r}")

    data = {}
    problems = []
    for name in FILES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            problems.append(f"{name}: missing or redirected")
            continue
        data[name] = path.read_bytes()
        if sha256(data[name]) != hashes[name]:
            problems.append(f"{name}: hash differs from PIN.json")
    if problems:
        raise ValueError("\n".join(problems))

    manifest = json.loads(data["manifest.json"])
    if not isinstance(manifest, dict) or not isinstance(manifest.get("output_sha256"), dict):
        raise TypeError("Missing scoring-package manifest file hashes")
    for name in FILES:
        if name != "manifest.json" and manifest["output_sha256"].get(name) != hashes[name]:
            raise ValueError(f"{name}: missing or different hash in package manifest")
    version = manifest.get("package_version")
    source = manifest.get("source_commit")
    if (
        not isinstance(version, str)
        or not version
        or not isinstance(source, str)
        or not source
        or pin.get("package_version") != version
        or pin.get("package_source_commit") != source
        or manifest.get("contract_version") != CONTRACT_VERSION
    ):
        raise ValueError("Scoring pin and package manifest disagree on version or source")

    if root.name.startswith("developer-v") and root.name != f"developer-v{version}":
        raise ValueError("Package directory disagrees with manifest version")

    version_lines = data["VERSION"].decode("utf-8").splitlines()
    if not version_lines or version_lines[0] != f"developer-v{version}":
        raise ValueError("VERSION disagrees with package manifest")
    for key, expected in (("contract_version", CONTRACT_VERSION), ("source_commit", source)):
        declared = [line for line in version_lines[1:] if line.startswith(key + "=")]
        if declared != [f"{key}={expected}"]:
            raise ValueError(f"VERSION disagrees with package manifest on {key}")
    contract = json.loads(data["developer_contract.json"])
    template = json.loads(data["selection_template.json"])
    run = json.loads(data["run_contract.json"])
    if (
        not isinstance(contract, dict)
        or contract.get("version") != CONTRACT_VERSION
        or not isinstance(template, dict)
        or template.get("contract_version") != CONTRACT_VERSION
    ):
        raise ValueError("Unsupported developer or selection contract version")
    if (
        not isinstance(contract.get("rows"), list)
        or not isinstance(run, dict)
        or not isinstance(run.get("approved_policies"), list)
    ):
        raise TypeError("Missing scoring rules or run policies")
    rule_ids = set()
    for row in contract["rows"]:
        rule_id = row.get("rule_id") if isinstance(row, dict) else None
        if not isinstance(rule_id, str) or not rule_id or rule_id in rule_ids:
            raise ValueError("Scoring rules require distinct rule IDs")
        rule_ids.add(rule_id)
    return pin, root, contract, run
