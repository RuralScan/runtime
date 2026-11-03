"""Check that the package files come from one version.

An update delivered as several files can be applied only in part, and the symptom is an error deep inside
a run. These checks look at the contracts between files (not at file dates), so they name what is out of date.
"""
from __future__ import annotations

import dataclasses
import inspect


def check() -> list[str]:
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
        need(hasattr(gee, "request_plan"), "data/gee.py is out of date (no request_plan)")
    except ImportError as exc:
        problems.append(f"data/gee.py: {exc}")

    for module, names in (("overview", ("make_overview",)), ("viz", ("plot_overview", "plot_combo_ladder", "plot_layers")),
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
    if problems:
        raise RuntimeError(
            "The package files are not all from the same update:\n  - " + "\n  - ".join(problems)
            + "\nCopy ALL the files of the latest update zip over the repository (not only some of them), "
              "then restart the kernel."
        )
