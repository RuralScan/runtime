import shutil
from pathlib import Path

from nbs_runtime.contract import UPSTREAM, verify_pin


def test_pin_is_intact():
    assert verify_pin() == []


def test_tampering_is_detected(tmp_path: Path):
    shutil.copytree(UPSTREAM, tmp_path / "upstream")
    pin = (tmp_path / "upstream" / "PIN.json").read_text(encoding="utf-8")
    assert "package" in pin
    target = next((tmp_path / "upstream").glob("*/*/run_contract.json"))
    target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert any("run_contract.json" in p for p in verify_pin(tmp_path / "upstream"))
