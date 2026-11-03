# Design

## Boundary

| | CIAT/nbs_ruralscan | this repository |
|---|---|---|
| Owns | Evidence, recipes (T0-T7), approved policies, generated packages | Data access, rasters, run execution, outputs, run manifests |
| Changes values? | Yes, through review and regeneration | Never |
| Publishes | Immutable `developer-vX.Y.Z` packages | Results and notebooks |

## What we inherit

| From CIAT | Used for | Where |
|---|---|---|
| `developer_contract.json` | Rows, function options (intervals), gate definitions | `upstream/.../` |
| `run_contract.json` | Approved policy: criteria, weights, aggregation, masks | `upstream/.../` |
| `conformance_examples.json`, `policy_conformance.json` | 40 response cases + 6 policy cases | `tests/test_conformance.py` |
| `manifest.json`, `VERSION` | Package version, source commit, file hashes | `upstream/.../` |
| Tables for later modules (T1, T2, T3, T5, T6, T6E, T7, lookups) | M2 to M5 | add to the allowlist when each module starts |

Not copied: the evidence registers, ingestion and QA code, dashboards, discovery logs,
historical packages. Read them on GitHub at the pinned commit when a number needs tracing.

## Updating from CIAT

A package version is immutable. A new version is a new folder (`developer-v0.1.6`), never an edit.

1. Find the new version in `handoff/agroforestry-t4/` on CIAT main and note the commit.
2. `python scripts/sync_upstream.py --ref <commit> --package agroforestry-t4/developer-vX.Y.Z`
   (refuses to pin if a file disagrees with CIAT's own manifest hashes).
3. `pytest`. Failing conformance cases or a fingerprint hold mean CIAT changed a function:
   read the package README and decide; do not patch values locally.
4. Commit `upstream/` with the new pin in one pull request.
5. Re-run the notebook. Every run records the pin, so results from different package
   versions stay distinguishable and reproducible.

Updating is deliberate on purpose: following `main` automatically would silently change
analytical results. Pin to released package versions only.

## Layout

```
upstream/        pinned CIAT package (read-only) + PIN.json
config/runs/     one TOML per run: AOI, grid, decisions, simulation, data order
src/nbs_runtime/ contract.py responses.py gates.py aggregate.py   CIAT's calculation per pixel
                 aoi.py runconfig.py plan.py                       what a run is, what blocks it
                 grid.py rasters.py firstpass.py                   CIAT's calculation over rasters
                 simulate.py compare.py viz.py                     ordinal simulation and maps (ours)
                 data/                                             synthetic (tests) and gee (Earth Engine)
scripts/         sync_upstream.py
tests/           conformance, pin integrity, policy, plan, aoi, config, simulation
notebooks/       01_agroforestry.ipynb: parameters cell, plan, simulation, saved results
docs/
```

## Run configuration

A run is one TOML file, optionally overridden by the notebook's parameters cell. Unknown keys are an
error (a typo cannot fall back to a default). Empty decisions stay empty and `plan()` reports them as holds.
Every run records a fingerprint of its configuration and of the AOI, next to the CIAT package pin, so two
results can be compared knowing exactly what differed.

The AOI is a bounding box for now. Naming a source (for example GADM 4.1) clears the hold; the box then
stays as a warning until the official boundary is fetched by the data adapter.

## Ordinal simulation

CIAT gives each class a score by convention. `simulate.py` keeps the order of the classes and the class
each pixel falls in, and redraws the scores many times in increasing order, so a pixel's index becomes a
median and a 5 to 95% range next to CIAT's own index. It does not replace CIAT's calculation.

- `scale_mode = "independent"` (default): every variable draws its own scale.
- `scale_mode = "shared"`: one scale per class label, common to all variables (wider ranges).
- `weight_concentration`: 0 keeps CIAT's weights fixed; a positive value lets them vary around them.
- Results are computed once per class combination (48 for the first pass) and a map is a lookup.
- Known limits: the draws use an uninformative prior (so the median sits below CIAT's convention); it
  quantifies the effect of the scale and weights only, not the origin of the curves or any validation.

## Layers, resolution and the cache

The first pass runs on a **250 m grid**. SoilGrids is native 250 m and soil is half of the index weight (carbon 25%,
pH 25%), so nothing finer is supported by the soil: a 30 m grid would repeat each soil value in blocks of about
8 x 8 cells and add no information (CIAT: resampling cannot add information). The finer layers are aggregated,
not resampled. Haiti's box is 936 x 1,292 cells at 250 m (1.2 million), against 84 million at 30 m.

`data/gee.py` follows CIAT's `DATASET_INPUTS.md`:

- slope in degrees is derived from SRTM at its native ~30 m scale and only then averaged to the grid;
- soil is SoilGrids at 0-5, 5-15 and 15-30 cm, kept as stored and converted once (dg/kg x 0.1; pH x 0.1), then
  weighted by layer thickness;
- land cover is stored as the **share of each WorldCover class** in every cell. Which classes count as host is decided
  afterwards from CIAT's gate, so a change in the gate does not need a new extraction.

Missing values stay unknown: nothing is filled with a mean.

Earth Engine's cost is set by the fine pixels it must read (about 712 million for WorldCover over Haiti's box), not by the
output resolution, so a coarse grid does not make the extraction faster. Requests are split into tiles to stay within
Earth Engine's per-user memory. Each extracted layer is saved at once in a `LayerCache` (`cache/layers/<key>/`, ignored
by git), keyed by the grid and the assets used. An interrupted run resumes, and later runs, or a new CIAT package
version, read the cache and never call Earth Engine again. The module needs an authenticated session and is not
tested offline; the notebook's quick check (section 3a) pulls a small box and maps every layer first.

To make the extraction cheaper later: use the official polygon instead of the bounding box (Haiti's land is about
28,000 km2 against 71,000 km2 for the box), and run large areas as batch exports.

## Two ways to run

| | Quick overview | Detailed |
|---|---|---|
| `RESOLUTION_M` | 5000 (47 x 65 cells for Haiti) | 250 |
| `HOST_GATE` | `"skip"` | `"apply"` |
| Earth Engine requests | about 54 | about 400 to 500 |

A 5 km grid is not faster by itself, because Earth Engine still reads every fine pixel under each cell; what makes the
quick overview quick is skipping WorldCover (the slowest layer) and running requests in parallel (`GEE_WORKERS`). Before
extracting, the notebook prints how many requests the run can make, and while it runs it shows the tiles done, the time spent
and a rough time left. Each grid has its own cache folder, so changing the resolution never reuses another grid's layers.

At 5 km the inputs are averaged over each cell before the functions are applied (a cell's mean slope, not the slope of each
30 m cell), so the result is an overview and not a substitute for the 250 m calculation.

## Overview and sensitivity

The calculation stays at the fine grid and only the **result** is summarised in coarse blocks (`overview.py`, for example 5 km):
each block's value is the mean over its scored cells, and a block with too few scored cells (`MIN_SCORED_SHARE`) stays
unknown. A block is never scored from an averaged input, because averaging a step function's input hides what is inside the
block (half 2 degrees, half 45 degrees averages to a slope the function leaves unknown). The notebook also draws a *ladder*:
for every class combination present in the data, CIAT's index and the simulated 5-95% range. Where two bars overlap, the order
of those classes is not settled by the data.

For a fast, exploratory view, `HOST_GATE = "skip"` never reads WorldCover (the slowest layer) and treats every cell as host;
`ONLY_FROM_CACHE = True` never calls Earth Engine. Both are labelled on every map and in the manifest.

Open question for CIAT: a cell's mean slope hides variation inside the cell (half flat, half steep gives a moderate
mean). Scoring each fine cell and averaging the scores would be more faithful; the aggregation method is to confirm.

`firstpass.py` computes CIAT's partial terrain-and-soil index over the rasters, and `compare.py` / `viz.py` put the
ordinal simulation next to it. Two modes: `policy` (every decision recorded; today every cell would be unknown
because the legal mask is not assessed) and `exploratory` (legal mask not applied, labelled on every map and in the
manifest). `data/synthetic.py` gives offline test data; its results are never Haiti.

Two readings of CIAT's text to confirm: a cell is host-eligible when any of its fine cells is (the eligible share is
kept as a layer), and a cell with no eligible host but some unknown host is unknown rather than excluded.

`data.order = ["local", "hf", "gee"]` records where a layer is looked for. The Hugging Face dataset (WorldClim,
aridity, HWSD v2, OSM roads) is not an input of this first pass. This is our reading of CIAT's brief, to confirm.

## Open decisions (owner)

- Haiti applicability of the India pools and the official AOI boundary and its source (method lead, Diego).
- Adopt or replace the SoilGrids 0-30 cm proposal; record a decision reference (method lead).
- Assessed legal mask; until then the policy returns "unknown" for every pixel.
- Host inputs, GEE adapters, band and unit contracts (Diego, Brayden).
