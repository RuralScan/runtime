"""A publisher switch preserves immutable packages and supports legacy pins."""

import json
import shutil

import pytest

from nbs_runtime._scoring_pin import FILES, sha256
from nbs_runtime.contract import UPSTREAM, load_package, verify_pin


@pytest.fixture
def publisher(monkeypatch):
    monkeypatch.syspath_prepend(str(UPSTREAM.parent / "scripts"))
    import sync_upstream

    pin = json.loads((UPSTREAM / "PIN.json").read_bytes())
    package = pin["package"]
    files = {name: (UPSTREAM / package / name).read_bytes() for name in FILES}
    urls = []

    def fetch(url):
        urls.append(url)
        if "/commits/" in url:
            return json.dumps({"sha": "a" * 40}).encode()
        if "/contents/" in url:
            return b'[{"name":"developer-v0.1.5","type":"dir"}]'
        return files[url.rsplit("/", 1)[1]]

    monkeypatch.setattr(sync_upstream, "_get", fetch)
    return sync_upstream, pin, files, urls


def test_switch_publisher_preserves_package_and_resolves_ref_once(publisher, tmp_path):
    tool, old, _, urls = publisher
    dest = tmp_path / "upstream"
    shutil.copytree(UPSTREAM, dest)
    first = load_package(dest)
    args = [
        "--repo",
        "example/publisher",
        "--source-root",
        "release/packages",
        "--upstream",
        str(dest),
    ]
    assert tool.main([*args, "--ref", "release/testing"]) == 0
    assert urls[0] == "https://api.github.com/repos/example/publisher/commits/release%2Ftesting"
    ref = "a" * 40
    prefix = f"https://raw.githubusercontent.com/example/publisher/{ref}/release/packages/"
    assert urls[1:] == [f"{prefix}{old['package']}/{name}" for name in FILES]
    second = load_package(dest)
    assert second.pin["ref"] == ref
    assert second.pin["source_repo"] == "example/publisher"
    assert second.pin["source_root"] == "release/packages"
    assert first.pin == old and second.pin["files"] == old["files"]
    assert first.contract == second.contract and first.run_contract == second.run_contract
    urls.clear()
    assert tool.main([*args, "--check"]) == 0 and not urls


def test_list_uses_selected_publisher_root_family_and_ref(publisher, capsys):
    tool, _, _, urls = publisher
    assert (
        tool.main(
            [
                "--list",
                "--repo",
                "example/publisher",
                "--source-root",
                "packages",
                "--package",
                "example-t4/developer-v0.1.5",
                "--ref",
                "release/testing",
            ]
        )
        == 0
    )
    assert urls == [
        "https://api.github.com/repos/example/publisher/contents/packages/example-t4?ref=release%2Ftesting"
    ]
    assert capsys.readouterr().out.strip() == "developer-v0.1.5"


@pytest.mark.parametrize(
    "fault", ["truncated", "missing_hash", "unsupported", "different_package", "duplicate_rules"]
)
def test_refusal_preserves_active_pin_and_bytes(publisher, tmp_path, fault, capsys):
    tool, pin, files, _ = publisher
    dest = tmp_path / "upstream"
    shutil.copytree(UPSTREAM, dest)
    original = (dest / "PIN.json").read_bytes()
    manifest = json.loads(files["manifest.json"])
    if fault == "truncated":
        files["run_contract.json"] = b"{"
        reason = "run_contract.json: missing or different hash"
    elif fault == "missing_hash":
        del manifest["output_sha256"]["run_contract.json"]
        reason = "run_contract.json: missing or different hash"
    elif fault == "unsupported":
        manifest["contract_version"] = "9.0.0"
        reason = "Unsupported scoring contract"
    elif fault == "duplicate_rules":
        contract = json.loads(files["developer_contract.json"])
        contract["rows"].append(contract["rows"][0])
        files["developer_contract.json"] = json.dumps(contract).encode()
        manifest["output_sha256"]["developer_contract.json"] = sha256(
            files["developer_contract.json"]
        )
        reason = "distinct rule IDs"
    else:
        reason = "Existing immutable package differs"
        files["developer_contract.json"] += b" "
        manifest["output_sha256"]["developer_contract.json"] = sha256(
            files["developer_contract.json"]
        )
    files["manifest.json"] = json.dumps(manifest).encode()
    assert tool.sync("a" * 40, pin["package"], repo="example/publisher", upstream=dest) == 1
    assert reason in capsys.readouterr().err
    assert (dest / "PIN.json").read_bytes() == original
    assert verify_pin(dest) == []


def test_download_interruption_leaves_first_install_unselected(publisher, monkeypatch, tmp_path):
    tool, pin, files, _ = publisher

    def interrupted(url):
        leaf = url.rsplit("/", 1)[1]
        if leaf == "manifest.json":
            raise OSError("connection lost")
        return files[leaf]

    monkeypatch.setattr(tool, "_get", interrupted)
    assert tool.sync(pin["ref"], pin["package"], upstream=tmp_path) == 1
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "field,value", [("contract_version", "9.0.0"), ("files", {}), ("package", "../other")]
)
def test_offline_reader_and_check_reject_invalid_pin(publisher, tmp_path, field, value):
    tool, pin, _, urls = publisher
    shutil.copytree(UPSTREAM, tmp_path, dirs_exist_ok=True)
    (tmp_path / "PIN.json").write_text(json.dumps({**pin, field: value}))
    assert tool.check(tmp_path) == 1
    with pytest.raises(RuntimeError):
        load_package(tmp_path)
    assert not urls
