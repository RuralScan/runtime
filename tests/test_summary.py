import pytest

np = pytest.importorskip("numpy")

from nbs_runtime.compare import compare
from nbs_runtime.data.synthetic import synthetic_layers
from nbs_runtime.firstpass import run_first_pass
from nbs_runtime.grid import Grid
from nbs_runtime.plan import plan
from nbs_runtime.runconfig import RunConfig
from nbs_runtime.simulate import build_level_space, simulate
from nbs_runtime.summary import (
    build_summary,
    describe_gap,
    function_gaps,
    gaps_by_variable,
)

POLICY = "planted_terrain_soil_v1"
GRID = Grid.from_bbox((-74.5, 18.0, -71.6, 20.1), 5000)


@pytest.fixture(scope="module")
def space(pkg):
    return build_level_space(pkg, pkg.policy(POLICY))


def test_gaps_of_a_made_up_function_are_found_exactly():
    ivs = [
        {"lower": None, "upper": 0.0, "lower_inclusive": False, "upper_inclusive": False, "score": 0.9},
        {"lower": 0.0, "upper": 5.0, "lower_inclusive": True, "upper_inclusive": False, "score": 0.5},
        {"lower": 5.0, "upper": 9.0, "lower_inclusive": False, "upper_inclusive": False, "score": None,
         "flags": ["median_tie"]},
        {"lower": 9.0, "upper": 12.0, "lower_inclusive": True, "upper_inclusive": True, "score": 0.1},
    ]
    gaps = function_gaps(ivs)
    assert [(g["lo"], g["hi"]) for g in gaps] == [(5.0, 9.0), (12.0, None)]
    assert "median tie" in gaps[0]["reason"] and "no source covers" in gaps[1]["reason"]


def test_adjacent_ranges_with_the_same_reason_are_merged():
    ivs = [
        {"lower": 0.0, "upper": 1.0, "lower_inclusive": False, "upper_inclusive": False, "score": None,
         "flags": ["exclusion_conflict"]},
        {"lower": 1.0, "upper": 2.0, "lower_inclusive": False, "upper_inclusive": False, "score": None,
         "flags": ["exclusion_conflict"]},
    ]
    assert [(g["lo"], g["hi"]) for g in function_gaps(ivs) if g["lo"] == 0.0] == [(0.0, 2.0)]


def test_the_real_gaps_in_ciats_functions(space):
    gaps = gaps_by_variable(space)
    assert [(g["lo"], g["hi"]) for g in gaps["slope"]] == [(20.0, 44.0)]
    assert gaps["soil_organic_carbon"] == []
    ph = [(g["lo"], g["hi"]) for g in gaps["soil_ph"]]
    assert (5.5, 6.5) in ph and (7.2, 7.5) in ph and (None, 4.6) in ph and (8.5, None) in ph
    text = describe_gap({"lo": 5.5, "hi": 6.5, "reason": "x"}, "pH")
    assert text.startswith("5.5 to 6.5 pH")


def test_every_gap_value_really_has_no_score(space):
    from nbs_runtime.responses import score_value

    for name, opt in zip(space.names, space.options, strict=True):
        for g in gaps_by_variable(space)[name]:
            lo = g["lo"] if g["lo"] is not None else g["hi"] - 2.0
            hi = g["hi"] if g["hi"] is not None else g["lo"] + 2.0
            assert score_value(opt["intervals"], (lo + hi) / 2) is None


def test_the_summary_reports_the_run_honestly(pkg, space):
    cfg = RunConfig.load(__import__("pathlib").Path(__file__).resolve().parents[1]
                         / "config" / "runs" / "agroforestry_planted_first_pass.toml",
                         {"grid": {"resolution_m": 5000}})
    lay = synthetic_layers(GRID, seed=3)
    lay.host_gate_applied = False
    fp = run_first_pass(space, lay, "exploratory")
    res = simulate(space, n_draws=300, seed=1)
    text = build_summary(cfg, pkg, plan(cfg, pkg), lay, fp, compare(fp, res), res, space, GRID, "synthetic")
    assert text.startswith("# Run summary: Haiti")
    for needle in ("Read this first", "exploratory", "host gate not applied", "SYNTHETIC", "placeholder box",
                   "Where the index can be computed", "Gaps in CIAT's functions", "5.5 to 6.5 pH",
                   "20 to 44 degrees", "Points to raise with CIAT", "Limits of this view"):
        assert needle in text, needle
    from nbs_runtime.firstpass import REASONS, explain_unscored

    in_functions = explain_unscored(space, lay)[REASONS[0]][0]
    assert f"**{in_functions:,} (" in text                         # the count in the text is the real one
    assert f"{int(fp.scored.sum()):,} cells are scored" not in text   # gate not applied: no second count
    assert "have all three values inside CIAT's functions" in text and "soil pH 25%" in text
