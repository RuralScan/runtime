"""Fetch an immutable scoring package from GitHub, or verify the local pin offline."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nbs_runtime._scoring_pin import (
    FILES,
    package_dir,
    publisher_root,
    read_pinned,
    relative_path,
    sha256,
)

REPO = "CIAT/nbs_ruralscan"
ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream"


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def _commit(repo: str, ref: str) -> str:
    if re.fullmatch(r"[0-9a-fA-F]{40}", ref):
        return ref.lower()
    result = json.loads(_get(f"https://api.github.com/repos/{repo}/commits/{quote(ref, safe='')}"))
    sha = result.get("sha") if isinstance(result, dict) else None
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError(f"Could not resolve {ref!r} to a commit in {repo}")
    return sha


def sync(ref: str, package: str, *, repo=REPO, source_root="handoff", upstream=UPSTREAM) -> int:
    try:
        source_root = publisher_root(repo, source_root)
        dest = package_dir(upstream, package)
        commit = _commit(repo, ref)
        base = f"https://raw.githubusercontent.com/{repo}/{commit}/{source_root}/{package}"
        upstream.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=upstream, prefix=".sync-") as scratch:
            stage = Path(scratch)
            pending = stage / package
            pending.mkdir(parents=True)
            hashes = {}
            for name in FILES:
                data = _get(f"{base}/{name}")
                (pending / name).write_bytes(data)
                hashes[name] = sha256(data)
            manifest = json.loads((pending / "manifest.json").read_bytes())
            if not isinstance(manifest, dict):
                raise TypeError("Invalid scoring-package manifest")
            pin = {
                "source_repo": repo,
                "ref": commit,
                "package": package,
                "package_version": manifest.get("package_version"),
                "contract_version": manifest.get("contract_version"),
                "package_source_commit": manifest.get("source_commit"),
                "pinned_on": date.today().isoformat(),
                "files": hashes,
            }
            if source_root != "handoff":
                pin["source_root"] = source_root
            pin_path = stage / "PIN.json"
            pin_path.write_text(json.dumps(pin, indent=2) + "\n", encoding="utf-8")
            read_pinned(stage)

            # Package directories are immutable. Expose a complete directory before
            # switching the selector, so existing readers can finish on their pin.
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                try:
                    pending.rename(dest)
                except OSError:
                    if not dest.is_dir():
                        raise
            if pending.exists():
                for name in FILES:
                    path = dest / name
                    if (
                        path.is_symlink()
                        or not path.is_file()
                        or sha256(path.read_bytes()) != hashes[name]
                    ):
                        raise ValueError(f"Existing immutable package differs: {package}/{name}")
            pin_path.replace(upstream / "PIN.json")
    except (OSError, ValueError, TypeError) as exc:
        print(f"Cannot pin scoring package: {exc}", file=sys.stderr)
        return 1
    print(f"Pinned {package} from {repo} @ {commit[:12]} ({len(FILES)} files)")
    return 0


def check(upstream=UPSTREAM) -> int:
    try:
        pin, *_ = read_pinned(upstream)
    except (OSError, ValueError, TypeError) as exc:
        print(f"Invalid scoring pin: {exc}", file=sys.stderr)
        return 1
    print(f"OK: {pin['package']} @ {pin['ref'][:12]}")
    return 0


def list_versions(*, repo=REPO, source_root="handoff", family="agroforestry-t4", ref="main") -> int:
    try:
        source_root = publisher_root(repo, source_root)
        path = f"{source_root}/{relative_path(family)}"
        url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={quote(ref, safe='')}"
        items = json.loads(_get(url))
        if not isinstance(items, list) or any(
            not isinstance(item, dict) or not isinstance(item.get("name"), str) for item in items
        ):
            raise ValueError(f"Expected a directory at {path}")
        versions = sorted(
            i["name"]
            for i in items
            if i.get("type") == "dir" and i["name"].startswith("developer-v")
        )
    except (OSError, ValueError, TypeError) as exc:
        print(f"Could not list scoring packages: {exc}", file=sys.stderr)
        return 1
    print(*versions, sep="\n")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ref", help="Publisher commit, tag or branch. Pins record the full commit SHA."
    )
    parser.add_argument("--package", default="agroforestry-t4/developer-v0.1.5")
    parser.add_argument("--repo", default=REPO, help="GitHub publisher as owner/repository")
    parser.add_argument(
        "--source-root", default="handoff", help="Package directory in the publisher repository"
    )
    parser.add_argument(
        "--upstream", type=Path, default=UPSTREAM, help="Local scoring-package store"
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="Verify the installed pin offline")
    modes.add_argument(
        "--list", action="store_true", help="List versions of the selected package family"
    )
    args = parser.parse_args(argv)
    if args.check:
        return check(args.upstream)
    if args.list:
        return list_versions(
            repo=args.repo,
            source_root=args.source_root,
            family=args.package.rsplit("/", 1)[0],
            ref=args.ref or "main",
        )
    if not args.ref:
        parser.error("--ref is required to pin")
    return sync(
        args.ref, args.package, repo=args.repo, source_root=args.source_root, upstream=args.upstream
    )


if __name__ == "__main__":
    sys.exit(main())
