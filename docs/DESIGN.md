# Design

## Boundary

| | Evidence skill (RuralScan/evidence) | this repository |
|---|---|---|
| Owns | Literature, the synthesis skill, the recipes (function + weight per variable) | Roles, datasets, the index, sensitivity, outputs |
| Changes recipe values? | Yes, by regenerating them | Never |
| Delivers | `config/recipes/<nbs>/<family>/<variable>.json` | Maps, rasters and a manifest per run |

A recipe file:

```json
{
  "practice": "agroforestry", "family": "cross_family", "variable": "slope", "unit": "degrees",
  "weight": 0.25, "shape": "linear",
  "x": [2.29, 2.86, ...], "score": [0.9, 0.85, ...],
  "low": [...], "high": [...],
  "valid_range": [null, null], "papers": ["chuma_2021", "haile_2024"], "upstream_commit": "..."
}
```

`recipes.py` validates every file: `x` strictly increasing, one score per node, scores in 0-1, `low <= score <= high`
when a band is given, `weight` a number (not a list), and the file sitting in the folder its `practice` and `family`
name. `upstream_commit` and the SHA-256 of each file are recorded in every run manifest.

## The index (approach 3)

```
S = L · G · M
L = min(z_j, j limiting)               no weights
G = 1 eligible land cover, 0 not       NaN where land cover is unknown
M = Σ w_j z_j, j modulating            recipe weights renormalised among the modulating variables
z_j = f_j(x_j)                         linear between the recipe's nodes
```

- **Roles** come from `config/roles.toml`, following the methodology's rule: what a project cannot change
  (climate) is limiting, what it can manage (soil, terrain, vegetation) is modulating. A limiting variable's
  recipe weight is ignored and `plan()` says so.
- **No hard limits** beyond eligibility. A limiting variable excludes only where its curve reaches 0; otherwise it
  caps the score (good terrain cannot lift a cell above its worst climate score).
- **Outside the evidence.** A value outside the first and last node has no evidence behind it: the cell is
  `OUTSIDE_EVIDENCE` (not evaluated). `outside = "clamp"` repeats the end-node score instead; it is recorded and
  labelled when used. The share of cells outside each recipe's evidence is reported in every run.
- **Unknown is never zero.** No data in any input, or unknown land cover, gives `NO_DATA`. Weights are never
  renormalised per pixel.
- **Families.** Each `<nbs>/<family>` folder is computed on its own. `aggregate_families` summarises several
  families per cell (mean with failing families counted as 0, breadth, quality), as the worked example does for
  species. The rule for one layer per practice is still to be agreed.

Pixel states: `SCORED`, `NOT_ELIGIBLE`, `NO_DATA`, `OUTSIDE_EVIDENCE` (precedence in that order, from
not eligible down). `Result` keeps L, G, M, every z and which limiting variable sets L, so any cell can be
explained step by step (`workflow.explain_pixel`).

**Classes.** Score 0 is its own class; positive scores are split into quintiles over the scored area (3 classes when
ties make quintiles inseparable, said in a note). Fixed limits can be passed instead. The absolute minimum, median
and maximum are always reported next to the classes.

## Sensitivity

Three tests that need nothing beyond the recipes:

1. **Equal weights**: the methodology's benchmark.
2. **Dirichlet**: draws of the modulating weights from Dirichlet(concentration × recipe weights). Per cell: median,
   5-95% range, probability of being in the top quintile. A cell in the top quintile in at least 80% of the draws
   is a robust hotspot. Only M changes, so L and G stay fixed and the draws are a matrix product; draw-level
   statistics use a sample of the scored cells.
3. **Leave one out**: the map without each variable in turn (weights renormalised), compared on the cells scored
   in both maps.

Recipe bands (`low`, `high`) are read and drawn when a recipe has them; the tests do not depend on them.

## Data

`variables.py` is the single catalog: id, unit, dataset, native resolution, scale and how it is derived. A recipe is
matched to its data by the variable id, and its unit must match the catalog's (`same_unit`).

- Earth Engine (`data/gee.py`): one image per variable, averaged to the grid in tiles that respect Earth Engine's
  memory limit. Slope is derived from the 30 m SRTM elevation before averaging; elevation is the mean of the 30 m
  values; NDVI is the median of one year of MODIS 16-day composites; land cover is the share of each WorldCover
  class in every cell.
- Files (`data/hf.py`): the Hugging Face dataset `dagudelo30/nbs-ruralscan-data` (WorldClim, aridity, HWSD, OSM
  roads), a local copy first, reading only the window over the area.
- Every Earth Engine layer is saved in a `LayerCache` (`cache/layers/<key>/`), keyed by the grid and the assets, so
  it is extracted once. Changing an asset or a parameter in the catalog gives a new key.

**Eligibility** (`eligibility.py`) reads the class shares: a cell is eligible when the eligible classes cover at
least `min_share` of it. Changing the classes never needs a new extraction. `mode = "skip"` treats every cell as
eligible, for a quick look, and is labelled.

## Resolution

The grid is set per run. A finer grid than the coarsest input adds no information for that input (WorldClim is
1 km, MODIS NDVI 250 m). Earth Engine's cost is set by the fine pixels it reads (30 m slope, 10 m land cover), not by
the output resolution. `overview.py` summarises a fine result in coarse blocks (mean of scored cells; thin blocks
stay unknown); a block is never scored from an averaged input.

## Layout

```
config/recipes/  the skill's recipes, one folder per <nbs>/<family>
config/roles.toml  limiting or modulating, per variable
config/runs/     one TOML per run
src/nbs_runtime/ recipes.py curves.py suitability.py sensitivity.py   the index
                 eligibility.py plan.py runconfig.py workflow.py     what a run is and its steps
                 variables.py grid.py aoi.py overview.py viz.py audit.py
                 data/  variables_loader gee hf cache boundaries synthetic
scripts/         demo_suitability.py (offline), check_gee.py (tiny-box check)
notebooks/       01_suitability (the workflow), 02_variable_audit (the variables)
tests/
```

## Open points

- The rule for going from families to one layer per practice (Pete).
- Recipes whose nodes do not cover the physical range (for example slope starting at 2.3°, leaving flat land
  unevaluated): extend the recipes, or agree on `clamp` for given variables.
- Whether elevation, as a temperature proxy, should be limiting.
- Recipes and datasets for mangrove and forest restoration.
