"""Pinned recipe generations use content-addressed canonical manifests."""

import os

import pytest

from nbs_runtime._recipe_store import read_generation
from recipe_store_fixture import canonical, record, write_manifest


def make_store(store, data=b"x"):
    contents = {"tables/source.csv": data, "reports/résumé.json": b"{}"}
    document = {"format_version": 1, "files": {k: record(v) for k, v in contents.items()}}
    release_id, generation = write_manifest(store, canonical(document), contents)
    return release_id, generation, contents


def test_pinned_read_never_uses_current_pointer(tmp_path):
    first, generation, expected = make_store(tmp_path)
    second, _, revised = make_store(tmp_path, b"second")
    (tmp_path / "current.json").write_text("invalid pointer")
    loaded = read_generation(tmp_path, first)
    assert loaded == expected and read_generation(tmp_path, second) == revised
    (generation / "files/tables/source.csv").write_bytes(b"subsequent modification")
    assert loaded == expected
    with pytest.raises(ValueError, match="source.csv"):
        read_generation(tmp_path, first)


@pytest.mark.parametrize("release_id", ["", None, 1, "x" * 64, "F" * 64, "../manifest"])
def test_requires_an_explicit_sha(tmp_path, release_id):
    with pytest.raises(ValueError, match="release ID"):
        read_generation(tmp_path, release_id)


@pytest.mark.parametrize(
    "name",
    ["../outside", "/absolute", "c:/x", "dir//x", "a/.", "a.", "x ", "a\\b", "nul.txt", "a|b"],
)
def test_rejects_nonportable_or_redirecting_logical_names(tmp_path, name):
    identifier, _ = write_manifest(
        tmp_path, canonical({"format_version": 1, "files": {name: record()}})
    )
    with pytest.raises(ValueError, match="Invalid recipe path"):
        read_generation(tmp_path, identifier)


@pytest.mark.parametrize("names", [("a", "a/file"), ("Dir/a", "dir/b"), ("é/a", "e\u0301/b")])
def test_rejects_ambiguous_logical_names(tmp_path, names):
    records = dict.fromkeys(names, record())
    identifier, _ = write_manifest(tmp_path, canonical({"format_version": 1, "files": records}))
    with pytest.raises(ValueError, match="Conflicting recipe path"):
        read_generation(tmp_path, identifier)


@pytest.mark.parametrize("version", [True, "1", 2])
def test_declines_unsupported_format(tmp_path, version):
    doc = {"format_version": version, "files": {"x": record()}}
    identifier, _ = write_manifest(tmp_path, canonical(doc), {"x": b"x"})
    with pytest.raises(ValueError, match="manifest format"):
        read_generation(tmp_path, identifier)


@pytest.mark.parametrize(
    "bad_record",
    [
        {},
        record() | {"size": True},
        record() | {"size": -1},
        record() | {"size": 1.0},
        record() | {"sha256": 3},
        record() | {"sha256": "0" * 63},
        record() | {"extraneous": True},
    ],
)
def test_validates_manifest_record_types(tmp_path, bad_record):
    doc = {"format_version": 1, "files": {"x": bad_record}}
    identifier, _ = write_manifest(tmp_path, canonical(doc), {"x": b"x"})
    with pytest.raises(ValueError, match="Invalid recipe file record"):
        read_generation(tmp_path, identifier)


@pytest.mark.parametrize("broken", ["duplicates", "reformatted", "wrong_version_key", "empty"])
def test_requires_canonical_manifest(tmp_path, broken):
    raw = canonical({"format_version": 1, "files": {"x": record()}})
    if broken == "duplicates":
        raw = raw.replace(b'"format_version":1', b'"format_version":1,"format_version":1')
    elif broken == "reformatted":
        raw = raw.strip()
    elif broken == "wrong_version_key":
        raw = raw.replace(b'"format_version":', b'"format_version":1,"alternate":')
    else:
        raw = canonical({"format_version": 1, "files": {}})
    identifier, _ = write_manifest(tmp_path, raw, {"x": b"x"})
    with pytest.raises(ValueError):
        read_generation(tmp_path, identifier)


@pytest.mark.parametrize(
    "defect",
    ["wrong_manifest", "different_content", "size", "missing", "extra", "extra_dir", "extra_top"],
)
def test_entire_generation_is_verified(tmp_path, defect):
    identifier, generation, _ = make_store(tmp_path)
    path = generation / "files/tables/source.csv"
    if defect == "wrong_manifest":
        (generation / "manifest.json").write_bytes(b"{}")
    elif defect == "different_content":
        path.write_bytes(b"y")
    elif defect == "size":
        # Matching digest with an incorrect declared byte count.
        doc = {"files": {"x": record() | {"size": 2}}, "format_version": 1}
        identifier, _ = write_manifest(tmp_path, canonical(doc), {"x": b"x"})
    elif defect == "missing":
        path.unlink()
    elif defect == "extra":
        path.with_name("unknown").touch()
    elif defect == "extra_dir":
        path.with_name("unlisted").mkdir()
    else:
        (generation / "unlisted").touch()
    with pytest.raises(ValueError):
        read_generation(tmp_path, identifier)


@pytest.mark.parametrize(
    "target",
    ["store", "generations", "generation", "files", "member_dir", "leaf", "manifest", "hardlink"],
)
def test_rejects_redirected_members_and_directories(tmp_path, target):
    store = tmp_path / "store"
    identifier, generation, _ = make_store(store)
    paths = {
        "store": store,
        "generations": generation.parent,
        "generation": generation,
        "manifest": generation / "manifest.json",
        "files": generation / "files",
        "member_dir": generation / "files/tables",
        "leaf": generation / "files/tables/source.csv",
        "hardlink": generation / "files/tables/source.csv",
    }
    member = paths[target]
    destination = tmp_path / "relocated"
    member.rename(destination)
    if target == "hardlink":
        os.link(destination, member)
    else:
        member.symlink_to(destination, target_is_directory=destination.is_dir())
    with pytest.raises(ValueError):
        read_generation(store, identifier)
