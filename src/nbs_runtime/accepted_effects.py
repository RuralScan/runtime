"""Read accepted AI scientific outputs without inferring or recalibrating effects."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _encoded(value):
    return (
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        + "\n"
    ).encode()


class AcceptedEffects:
    """Verified immutable values. Context selection is explicit, never inferred."""

    def __init__(self, root: Path, manifest_sha256: str):
        root = Path(root)
        _require(
            re.fullmatch(r"[a-f0-9]{64}", manifest_sha256),
            "Supply the independently trusted manifest SHA-256",
        )
        member = root / "manifest.json"
        _require(
            not root.is_symlink() and not member.is_symlink(), "Linked package is not permitted"
        )
        raw = member.read_bytes()
        _require(_digest(raw) == manifest_sha256, "Manifest checksum differs")
        manifest = json.loads(raw)
        _require(
            manifest.get("format") == "ruralscan.accepted-effects.v1"
            and manifest.get("scientific_synthesis") == "ai_authored_human_accepted",
            "Unsupported scientific release",
        )
        files = manifest.get("files", {})
        _require(isinstance(files, dict) and bool(files), "Empty accepted release")
        self._rows = {}
        for name, expected in files.items():
            match = re.fullmatch(r"([a-z][a-z0-9_-]*)/(T3|T6)\.json", name)
            _require(match, "Invalid accepted table path")
            path = root / name
            _require(
                not path.is_symlink() and not path.parent.is_symlink(),
                "Linked table is not permitted",
            )
            data = path.read_bytes()
            _require(_digest(data) == expected, "Accepted table checksum differs")
            rows = json.loads(data)
            _require(isinstance(rows, list), "Empty accepted table")
            for row in rows:
                key = (match[1], match[2], row["record_id"])
                _require(
                    key not in self._rows and row["nbs_id"] == match[1],
                    "Duplicate or mis-scoped row",
                )
                self._rows[key] = row
        actual = {
            p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()
        }
        _require(actual == {"manifest.json", *files}, "Unexpected package inventory")
        self._provenance = {}
        for entry in manifest.get("provenance", []):
            key = (entry["practice"], entry["table"], entry["record_id"])
            _require(key not in self._provenance, "Invalid acceptance inventory")
            _require(
                entry["status"] == "accepted"
                and entry.get("receipt")
                and re.fullmatch(r"[a-f0-9]{64}", entry.get("proposal_sha256", "")),
                "Row lacks scientific acceptance",
            )
            if entry.get("action") == "withdraw":
                _require(
                    key not in self._rows
                    and entry["row_sha256"] is None
                    and re.fullmatch(r"[a-f0-9]{64}", entry.get("previous_sha256", "")),
                    "Invalid withdrawal",
                )
            else:
                _require(
                    key in self._rows and entry["row_sha256"] == _digest(_encoded(self._rows[key])),
                    "Accepted row differs from scientific decision",
                )
            self._provenance[key] = entry
        _require(
            set(self._rows)
            == {k for k, v in self._provenance.items() if v.get("action") != "withdraw"},
            "Unaccepted row in package",
        )
        self.manifest_sha256 = manifest_sha256

    def row(self, practice: str, table: str, record_id: str) -> dict:
        """Return precisely the chosen accepted row; no nearest-context fallback."""
        key = (practice, table, record_id)
        _require(key in self._rows, "No accepted output for the requested record")
        return json.loads(json.dumps(self._rows[key]))

    def provenance(self, practice: str, table: str, record_id: str) -> dict:
        key = (practice, table, record_id)
        _require(key in self._provenance, "No acceptance for the requested record")
        return dict(self._provenance[key])
