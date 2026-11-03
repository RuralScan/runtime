import pytest

pytest.importorskip("numpy")

from nbs_runtime import selfcheck
from nbs_runtime.data import assemble


def test_the_package_is_in_sync():
    assert selfcheck.check() == []
    selfcheck.assert_in_sync()


def test_an_old_assemble_is_detected_with_a_clear_message(monkeypatch):
    def old_assemble(slope, soc, ph, shares, groups, notes=()):
        return list(shares.values())                       # what the old file did with shares=None

    monkeypatch.setattr(assemble, "assemble_layers", old_assemble)
    problems = selfcheck.check()
    assert any("assemble.py is out of date" in p for p in problems)
    with pytest.raises(RuntimeError, match="Copy ALL the files"):
        selfcheck.assert_in_sync()


def test_a_missing_load_layers_option_is_detected(monkeypatch):
    from nbs_runtime.data import gee

    monkeypatch.setattr(gee, "load_layers", lambda grid, groups, cache=None: None)
    assert any("gee.py is out of date" in p for p in selfcheck.check())
