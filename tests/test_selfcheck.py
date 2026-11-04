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


# ---- stale Python session vs files out of date ----

def test_a_fresh_session_is_not_stale():
    assert selfcheck.session_is_stale() is False


def test_files_newer_than_the_import_mean_a_stale_session(monkeypatch):
    import nbs_runtime

    monkeypatch.setattr(nbs_runtime, "_IMPORTED_AT", 0.0)
    assert selfcheck.session_is_stale() is True
    assert selfcheck.check() == [selfcheck.STALE_MESSAGE]
    with pytest.raises(RuntimeError, match="Restart the kernel"):
        selfcheck.assert_in_sync()


def test_a_session_that_never_stamped_the_import_is_stale_when_the_disk_is_new(monkeypatch):
    import nbs_runtime

    monkeypatch.delattr(nbs_runtime, "_IMPORTED_AT")
    assert selfcheck.session_is_stale() is True


def test_an_old_init_on_disk_is_a_files_problem_not_a_session_problem(tmp_path):
    old_init = tmp_path / "__init__.py"
    old_init.write_text('__version__ = "0.0.1"\n', encoding="utf-8")        # no import stamp
    assert selfcheck.session_is_stale(old_init) is False
