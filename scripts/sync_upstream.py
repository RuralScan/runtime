"""Pin a generated T4 developer package from CIAT/nbs_ruralscan into ./upstream.

Only an allowlist of machine-readable files is copied (about 1 MB). A package is
immutable upstream, so a pin is (repository, commit, package path) plus file hashes.

  python scripts/sync_upstream.py --ref <commit> --package agroforestry-t4/developer-v0.1.5
  python scripts/sync_upstream.py --check          # verify files against PIN.json
  python scripts/sync_upstream.py --list           # list package versions on CIAT main
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

REPO = "CIAT/nbs_ruralscan"
ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream"
ALLOWLIST = [
    "developer_contract.json",
    "run_contract.json",
    "conformance_examples.json",
    "policy_conformance.json",
    "selection_template.json",
    "manifest.json",
    "VERSION",
]


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sync(ref: str, package: str) -> int:
    dest = UPSTREAM / package
    dest.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/{REPO}/{ref}/handoff/{package}"
    files, blobs = {}, {}
    for name in ALLOWLIST:
        blobs[name] = _get(f"{base}/{name}")
        files[name] = _sha(blobs[name])
    # Cross-check against the hashes CIAT published inside the package itself.
    manifest = json.loads(blobs["manifest.json"])
    problems = [
        f"{n}: differs from the upstream manifest"
        for n, h in manifest.get("output_sha256", {}).items()
        if n in files and n != "manifest.json" and files[n] != h
    ]
    if problems:
        print("REFUSING to pin:", *problems, sep="\n  ")
        return 1
    for name, data in blobs.items():
        (dest / name).write_bytes(data)
    pin = {
        "source_repo": REPO,
        "ref": ref,
        "package": package,
        "package_version": manifest.get("package_version"),
        "contract_version": manifest.get("contract_version"),
        "package_source_commit": manifest.get("source_commit"),
        "pinned_on": date.today().isoformat(),
        "files": files,
    }
    (UPSTREAM / "PIN.json").write_text(json.dumps(pin, indent=2) + "\n", encoding="utf-8")
    print(f"Pinned {package} @ {ref[:12]} ({len(files)} files)")
    return 0


def check() -> int:
    pin_path = UPSTREAM / "PIN.json"
    if not pin_path.exists():
        print("No upstream/PIN.json")
        return 1
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    bad = []
    for name, h in pin["files"].items():
        p = UPSTREAM / pin["package"] / name
        if not p.exists() or _sha(p.read_bytes()) != h:
            bad.append(name)
    if bad:
        print("Pinned files changed or missing:", *bad, sep="\n  ")
        return 1
    print(f"OK: {pin['package']} @ {pin['ref'][:12]}")
    return 0


def list_versions() -> int:
    url = f"https://api.github.com/repos/{REPO}/contents/handoff/agroforestry-t4?ref=main"
    try:
        items = json.loads(_get(url))
    except (OSError, ValueError) as exc:  # anonymous API calls are rate-limited
        print(f"Could not list versions ({exc}). Browse "
              f"https://github.com/{REPO}/tree/main/handoff/agroforestry-t4 instead.")
        return 1
    print(*sorted(i["name"] for i in items if i["name"].startswith("developer-v")), sep="\n")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ref", help="CIAT commit SHA to pin")
    ap.add_argument("--package", default="agroforestry-t4/developer-v0.1.5")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.check:
        sys.exit(check())
    if a.list:
        sys.exit(list_versions())
    if not a.ref:
        ap.error("--ref is required to pin")
    sys.exit(sync(a.ref, a.package))
