"""Check that the package files come from one version.

An update delivered as several files can be applied only in part, and the symptom is an error deep inside
a run. These checks look at the contracts between files (not at file dates), so they name what is out of date.
"""
from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path

STALE_MESSAGE = (
    "This Python session loaded an older version of the package before the files were updated. "
    "The files on disk are newer than what is running."
)


def session_is_stale(init_path: Path | None = None, tolerance_s: float = 5.0) -> bool:
    """True when the files on disk are newer than the code this Python session imported.

    Python does not reload a module it already imported, so copying new files over an open notebook has no
    effect until the kernel restarts. If the __init__.py on disk is itself old (it has no import stamp), the
    problem is the files, not the session.
    """
    import nbs_runtime

    init = Path(init_path) if init_path else Path(nbs_runtime.__file__)
    if "_IMPORTED_AT" not in init.read_text(encoding="utf-8"):
        return False
    imported = getattr(nbs_runtime, "_IMPORTED_AT", None)
    if imported is None:
        return True                                   # disk is new, memory is old
    newest = max(p.stat().st_mtime for p in init.parent.rglob("*.py"))
    return newest > imported + tolerance_s


def check() -> list[str]:
    if session_is_stale():
        return [STALE_MESSAGE]
    problems: list[str] = []

    def need(ok: bool, message: str) -> None:
        if not ok:
            problems.append(message)

    try:
        from .firstpass import Layers
        need("host_gate_applied" in {f.name for f in dataclasses.fields(Layers)},
             "firstpass.py is out of date (Layers has no host_gate_applied)")
    except ImportError as exc:
        problems.append(f"firstpass.py: {exc}")

    try:
        from .firstpass import STATUS
        need(5 in STATUS, "firstpass.py is out of date (no 'no data' state)")
    except ImportError as exc:
        problems.append(f"firstpass.py: {exc}")

    try:
        import numpy as np

        from .data.assemble import assemble_layers
        z = {d: np.zeros((2, 2)) for d in ("0-5", "5-15", "15-30")}
        try:
            assemble_layers(np.zeros((2, 2)), z, z, None, {"eligible": [], "ineligible": [], "unknown": []})
        except (AttributeError, TypeError) as exc:
            problems.append(f"data/assemble.py is out of date (cannot run without land-cover shares): {exc}")
    except ImportError as exc:
        problems.append(f"data/assemble.py: {exc}")

    try:
        from .data import gee
        params = inspect.signature(gee.load_layers).parameters
        need({"host", "cache_only", "workers"} <= set(params), "data/gee.py is out of date (load_layers parameters)")
        need(hasattr(gee, "request_plan") and hasattr(gee, "resolve_project"), "data/gee.py is out of date (no request_plan / resolve_project)")
    except ImportError as exc:
        problems.append(f"data/gee.py: {exc}")

    for module, names in (("overview", ("make_overview",)), ("summary", ("build_summary", "function_gaps")),
                          ("variables", ("CATALOG", "select", "contract_roles")), ("audit", ("audit_data", "summary_rows", "describe")),
                          ("data.hf", ("raster_source", "read_raster_to_grid", "road_distance_km")),
                          ("data.variables_loader", ("load_variable", "LoadContext")), ("firstpass", ("reason_map", "explain_unscored")),
                          ("viz", ("plot_overview", "plot_combo_ladder", "plot_layers", "plot_reasons", "plot_variable_audit", "plot_group_overview")),
                          ("grid", ("Grid",)), ("rasters", ("host_fractions_from_shares", "WORLDCOVER_CODES"))):
        try:
            mod = __import__(f"{__package__}.{module}", fromlist=["x"])
            for name in names:
                need(hasattr(mod, name), f"{module}.py is out of date (no {name})")
        except ImportError as exc:
            problems.append(f"{module}.py is missing or broken: {exc}")
    return problems


def assert_in_sync() -> None:
    problems = check()
    if problems == [STALE_MESSAGE]:
        raise RuntimeError(
            STALE_MESSAGE + "\nRestart the kernel (Kernel > Restart) and run the notebook again from the top."
        )
    if problems:
        raise RuntimeError(
            "The package files are not all from the same update:\n  - " + "\n  - ".join(problems)
            + "\nCopy ALL the files of the latest update zip over the repository (not only some of them), "
              "then restart the kernel."
        )
