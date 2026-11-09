"""Check the public pin, snapshot accessors and inspection command together."""

import json
from importlib.resources import files

import pytest

from nbs_runtime.recipe_release import load_recipe_release, main
from recipe_store_fixture import canonical, record, write_manifest


@pytest.fixture
def pinned(tmp_path, recipe_members):
    manifest = {
        "format_version": 1,
        "files": {name: record(data) for name, data in recipe_members.items()},
    }
    release_id, generation = write_manifest(tmp_path, canonical(manifest), recipe_members)
    pin = tmp_path / "pin.json"
    pin.write_text(
        json.dumps(
            {
                "release_id": release_id,
                "source_revision": "1" * 40,
                "practices": ["example"],
            }
        )
    )
    return pin, generation


def test_pinned_snapshot(tmp_path, pinned):
    pin, generation = pinned
    (tmp_path / "current.json").write_text("invalid mutable pointer")
    release = load_recipe_release(tmp_path, pin=pin)
    assert release.release_id == generation.name
    assert release.practices == ("example",)
    release.table("example", "T4").clear()
    assert len(release.table("example", "T4")) == 4
    assert release.table("example", "T6E") == []
    release.provenance.clear()
    assert release.provenance["source_revision"] == "1" * 40
    assert b"Example" in release.setting("example", "T0_nbs_registry.csv")
    release.report("example", "T4_readiness.json").clear()
    assert release.report("example", "T4_readiness.json")["execution_ready"] is False

    (generation / "files/settings/example/T0_nbs_registry.csv").write_bytes(b"changed")
    assert b"Example" in release.setting("example", "T0_nbs_registry.csv")
    with pytest.raises(ValueError, match="T0_nbs_registry"):
        load_recipe_release(tmp_path, pin=pin)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"release_id": "0" * 64}, "00000000"),
        ({"release_id": "latest"}, "Invalid release_id"),
        ({"source_revision": "2" * 40}, "source revision differs"),
        ({"practices": ["other"]}, "practice inventory"),
        ({"practices": ["example", "example"]}, "Duplicate practice"),
    ],
)
def test_pin_mismatch_stops_inspection(tmp_path, pinned, change, message, capsys):
    pin, _ = pinned
    pin.write_text(json.dumps(json.loads(pin.read_text()) | change))
    with pytest.raises(SystemExit) as stopped:
        main(["--store", str(tmp_path), "--pin", str(pin)])
    assert stopped.value.code == 1
    output = capsys.readouterr()
    assert message in output.err
    assert output.out == ""


def test_inspection_output(tmp_path, pinned, capsys):
    pin, generation = pinned
    assert main(["--store", str(tmp_path), "--pin", str(pin)]) == 0
    output = capsys.readouterr().out
    assert f"Release {generation.name}" in output
    assert "example: T3=1 T4=4 T6=1 T6E=0  execution_ready=false" in output


def test_bundled_pin_selects_its_named_generation(tmp_path):
    pin = json.loads(files("nbs_runtime").joinpath("recipe_pin.json").read_text(encoding="utf-8"))
    (tmp_path / "generations").mkdir()
    with pytest.raises(FileNotFoundError) as failed:
        load_recipe_release(tmp_path)
    assert failed.value.filename == str(tmp_path.resolve() / "generations" / pin["release_id"])
