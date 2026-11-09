"""Verify and read the files named by a v1 local recipe generation."""

from __future__ import annotations

import hashlib
import json
import ntpath
import os
import re
import stat
import unicodedata
from pathlib import Path, PureWindowsPath

_SHA = re.compile(r"[a-f0-9]{64}")
_BAD_COMPONENT = re.compile(r'[\x00-\x1f<>:"\\|?*]')


def _inventory(manifest: bytes, release_id: str) -> dict:
    if hashlib.sha256(manifest).hexdigest() != release_id:
        raise ValueError(f"Manifest fingerprint differs from pinned release {release_id}")

    data = json.loads(manifest)
    if not isinstance(data, dict) or set(data) != {"files", "format_version"}:
        raise ValueError("Invalid recipe manifest")
    if type(data["format_version"]) is not int or data["format_version"] != 1:
        raise ValueError("Unsupported recipe manifest format")
    files = data["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("Expected a nonempty recipe file inventory")
    for name, record in files.items():
        if (
            not isinstance(record, dict)
            or set(record) != {"size", "sha256"}
            or type(record["size"]) is not int
            or record["size"] < 0
            or not isinstance(record["sha256"], str)
            or not _SHA.fullmatch(record["sha256"])
        ):
            raise ValueError(f"Invalid recipe file record for {name!r}")

    # A v1 pin hashes sorted, compact ASCII JSON with one trailing newline.
    # Re-encoding also rejects a second spelling of the same manifest or duplicate keys.
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    if manifest != (canonical + "\n").encode("ascii"):
        raise ValueError("Recipe manifest is not in canonical v1 format")
    return files


def _reserved_component(name: str) -> bool:
    if hasattr(ntpath, "isreserved"):
        return ntpath.isreserved(name)
    return PureWindowsPath(name).is_reserved()  # Python 3.11 and 3.12


def _member_paths(inventory: dict) -> set[str]:
    """Return the complete expected tree, including directories, in portable spelling."""
    expected: set[str] = set()
    spellings: dict[str, str] = {}
    for name in inventory:
        parts = name.split("/")
        for index, part in enumerate(parts, start=1):
            if (
                not part
                or part in {".", ".."}
                or part.endswith((".", " "))
                or _BAD_COMPONENT.search(part)
                or _reserved_component(part)
            ):
                raise ValueError(f"Invalid recipe path {name!r}")
            path = "/".join(parts[:index])
            folded = unicodedata.normalize("NFC", path).casefold()
            if spellings.setdefault(folded, path) != path or (
                index < len(parts) and path in inventory
            ):
                raise ValueError(f"Conflicting recipe path {path!r}")
            expected.add(path)
    return expected


def _file_bytes(path: Path) -> bytes:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or path.resolve() != path:
        raise ValueError(f"Expected an independent regular file at {path}")
    return path.read_bytes()


def read_generation(store: Path, release_id: str) -> dict[str, bytes]:
    """Load verified bytes for an explicit release ID without following the current pointer."""
    if not isinstance(release_id, str) or not _SHA.fullmatch(release_id):
        raise ValueError("Specify a lowercase, 64-character SHA-256 recipe release ID")
    root = Path(store).absolute()
    if root.is_symlink() or (
        ".." in root.parts and root.resolve() != Path(os.path.abspath(root)).resolve()
    ):
        raise ValueError(f"Redirected recipe store: {root}")
    generation = root.resolve() / "generations" / release_id
    for directory in (generation.parent, generation, generation / "files"):
        if not stat.S_ISDIR(directory.lstat().st_mode) or directory.resolve() != directory:
            raise ValueError(f"Expected a recipe directory at {directory}")
    if {path.name for path in generation.iterdir()} != {"files", "manifest.json"}:
        raise ValueError(f"Unexpected entries in recipe generation {release_id}")

    inventory = _inventory(_file_bytes(generation / "manifest.json"), release_id)
    expected = _member_paths(inventory)
    files = generation / "files"
    actual = {path.relative_to(files).as_posix() for path in files.rglob("*")}
    if expected != actual:
        raise ValueError(f"Recipe generation {release_id} does not match its inventory")

    verified = {}
    for name, record in inventory.items():
        payload = _file_bytes(files / name)
        if (
            len(payload) != record["size"]
            or hashlib.sha256(payload).hexdigest() != record["sha256"]
        ):
            raise ValueError(f"Recipe member failed verification: {name}")
        verified[name] = payload
    return verified
