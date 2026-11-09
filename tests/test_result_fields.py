"""The public result names can change without invalidating existing runs or callers."""

import pickle
from dataclasses import asdict, fields, replace

import numpy as np
import pytest

from nbs_runtime.compare import Comparison
from nbs_runtime.firstpass import FirstPass
from nbs_runtime.overview import Overview


@pytest.mark.parametrize("cls,legacy", [(FirstPass, "pete"), (Overview, "ciat")])
def test_raster_keyword_attribute_and_saved_state(cls, legacy):
    raster = np.array([[0.4, np.nan]])
    args = (
        {
            "status": np.array([[0, 4]]),
            "combo_id": np.array([[1, -1]]),
            "eligible_fraction": np.ones((1, 2)),
            "mode": "exploratory",
        }
        if cls is FirstPass
        else {
            "factor": 2,
            "resolution_m": 500,
            "extent": (0, 1, 0, 1),
            "min_scored_share": 0.5,
            "median": raster,
            "p05": raster,
            "p95": raster,
            "p_top": raster,
            "coverage": raster,
        }
    )
    modern = cls(**args, baseline_index=raster)
    old = cls(**args, **{legacy: raster})
    for obj in (modern, old):
        assert obj.baseline_index is getattr(obj, legacy) is raster
        assert vars(obj)[legacy] is raster and "baseline_index" not in vars(obj)
        assert legacy in {f.name for f in fields(obj)}
        np.testing.assert_array_equal(asdict(obj)[legacy], raster)

    # Original positional argument order, including optional FirstPass fields.
    positional = cls(*(getattr(old, f.name) for f in fields(old)))
    assert positional.baseline_index is raster
    changed = raster.copy()
    setattr(old, legacy, changed)
    assert old.baseline_index is changed
    old.baseline_index = raster
    assert getattr(old, legacy) is raster
    assert replace(old, **{legacy: changed}).baseline_index is changed
    assert replace(old, baseline_index=changed).baseline_index is changed
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(modern)).baseline_index, raster)


def test_legacy_comparison_dictionary_remains_live():
    legacy = {"n_scored": 3, "spearman_pete_vs_median": 0.6, "caller_note": "demo"}
    layer = np.ones((1, 1))
    comp = Comparison(layer, layer, layer, layer, stats=legacy)
    assert comp.stats is legacy
    assert comp.statistics == {
        "n_scored": 3,
        "spearman_baseline_vs_median": 0.6,
        "caller_note": "demo",
    }
    legacy["spearman_pete_vs_median"] = 0.75
    assert comp.statistics["spearman_baseline_vs_median"] == 0.75
    empty = replace(comp, stats={"n_scored": 0, "n_cells": 1})
    assert empty.statistics == empty.stats == {"n_scored": 0, "n_cells": 1}
