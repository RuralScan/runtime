"""Accepted values are consumed verbatim; invalid or missing acceptance fails closed."""

import hashlib
import json

import pytest

from nbs_runtime.accepted_effects import AcceptedEffects


def encoded(value):
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def package(tmp_path):
    row = {
        "nbs_id": "practice",
        "record_id": "effect",
        "effect_direction": "moderate_positive",
        "evidence_level": "robust",
        "agreement_level": "high",
        "effect_confidence": "low",
        "justification": {"rationale": "Synthetic reviewed judgment; not a scientific result."},
    }
    path = tmp_path / "practice/T6.json"
    path.parent.mkdir()
    path.write_bytes(encoded([row]))
    manifest = {
        "format": "ruralscan.accepted-effects.v1",
        "scientific_synthesis": "ai_authored_human_accepted",
        "files": {"practice/T6.json": digest(path.read_bytes())},
        "provenance": [
            {
                "practice": "practice",
                "table": "T6",
                "record_id": "effect",
                "status": "accepted",
                "receipt": 42,
                "proposal_sha256": "a" * 64,
                "row_sha256": digest(encoded(row)),
            }
        ],
    }
    return tmp_path, row, manifest


def publish(package):
    root, _, manifest = package
    data = encoded(manifest)
    (root / "manifest.json").write_bytes(data)
    return AcceptedEffects(root, digest(data))


def test_values_survive_without_scoring_or_matrix(package):
    reader = publish(package)
    row = reader.row("practice", "T6", "effect")
    assert row == package[1]
    assert row["effect_confidence"] == "low"
    row["justification"]["rationale"] = "mutated by caller"
    assert reader.row("practice", "T6", "effect") == package[1]
    assert reader.provenance("practice", "T6", "effect")["receipt"] == 42
    with pytest.raises(ValueError, match="No accepted output"):
        reader.row("practice", "T6", "historical-fallback")


def test_requires_independent_manifest_pin(package):
    publish(package)
    with pytest.raises(ValueError, match="checksum"):
        AcceptedEffects(package[0], "b" * 64)
    with pytest.raises(ValueError, match="independently trusted"):
        AcceptedEffects(package[0], "")


@pytest.mark.parametrize("status", ["stale", "held", "historical_baseline"])
def test_unaccepted_status_is_rejected(package, status):
    package[2]["provenance"][0]["status"] = status
    with pytest.raises(ValueError, match="scientific acceptance"):
        publish(package)


def test_missing_provenance_and_changed_values_fail(package):
    package[2]["provenance"] = []
    with pytest.raises(ValueError, match="Unaccepted row"):
        publish(package)


def test_table_and_row_tampering_fail(package):
    reader = publish(package)
    path = package[0] / "practice/T6.json"
    path.write_bytes(encoded([dict(package[1], effect_confidence="high")]))
    with pytest.raises(ValueError, match="table checksum"):
        AcceptedEffects(package[0], reader.manifest_sha256)
    package[2]["files"]["practice/T6.json"] = digest(path.read_bytes())
    with pytest.raises(ValueError, match="differs from scientific decision"):
        publish(package)


def test_extra_files_and_path_escape_fail(package):
    (package[0] / "extra.json").write_text("[]")
    with pytest.raises(ValueError, match="inventory"):
        publish(package)
    package[2]["files"] = {"../escape.json": "a" * 64}
    with pytest.raises(ValueError, match="table path"):
        publish(package)


def test_withdrawal_is_visible_but_never_a_value(package):
    root, _, manifest = package
    path = root / "practice/T6.json"
    path.write_bytes(encoded([]))
    manifest["files"]["practice/T6.json"] = digest(path.read_bytes())
    manifest["provenance"][0].update(
        action="withdraw",
        previous_sha256="b" * 64,
        row_sha256=None,
    )
    reader = publish(package)
    assert reader.provenance("practice", "T6", "effect")["action"] == "withdraw"
    with pytest.raises(ValueError, match="No accepted output"):
        reader.row("practice", "T6", "effect")


def test_symlink_rejected(package):
    root, _, manifest = package
    table = root / "practice/T6.json"
    original = root / "target.json"
    table.rename(original)
    table.symlink_to(original)
    (root / "manifest.json").write_bytes(encoded(manifest))
    with pytest.raises(ValueError, match="Linked table"):
        AcceptedEffects(root, digest(encoded(manifest)))
