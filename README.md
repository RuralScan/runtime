# nbs-ruralscan-runtime

Runtime for **NbS Rural Scan**. It reads the generated T4 developer package published by
[CIAT/nbs_ruralscan](https://github.com/CIAT/nbs_ruralscan) (methodology and evidence live
there) and turns it into spatial results. This repository holds no methodology.

Current scope: the approved **planted-silvoarable first pass** (agroforestry): slope 50%,
soil organic carbon 25%, pH 25%, plus a host gate and a legal-exclusion mask. It is a
partial terrain-and-soil index, not complete agroforestry suitability.

## Read a six-practice recipe release

Install the runtime with `pip install .`. With an authorised copy of the release store on disk, run the inspection command.

```bash
python -m nbs_runtime.recipe_release --store /path/to/recipe-releases
```

The command verifies the files against the bundled [recipe pin](src/nbs_runtime/recipe_pin.json) and prints the source revision, each practice's table counts and its spatial execution status. The store must contain `generations/<release_id>/manifest.json` and the matching `files/` tree. A copy of that generation alone is sufficient. The directory can be read-only.

```python
from pathlib import Path
from nbs_runtime.recipe_release import load_recipe_release

release = load_recipe_release(Path("/path/to/recipe-releases"))
wetland_effects = release.table("wetland_management", "T6")
water_review = release.report("water_harvesting_conservation", "T4_readiness.json")
print(release.release_id, len(wetland_effects), water_review["implementation_readiness"])
```

Each practice supplies its T0 definition and authored settings, T3, T4, T6 and T6E tables, and synthesis and readiness reports. An empty table is a valid result. A held T4 mapping documents a reviewed decision and has no selectable scoring function. Consult the readiness report before interpreting its rows. The spatial first-pass notebook uses the separate developer package described below.

To inspect a different version, pass `--pin /path/to/trusted-pin.json` or the Python keyword `pin=Path(...)`. Obtain that pin independently of the supplied release files. It specifies the release ID, source revision and complete practice list. Keep the pin alongside the analysis that used it.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/sync_upstream.py --check     # pinned CIAT files are intact
pytest                                      # CIAT's 40 + 6 cases, plus our own checks
```

```python
from pathlib import Path
from nbs_runtime.plan import plan

p = plan(Path("config/runs/agroforestry_planted_first_pass.toml"))
print(p.package_version, p.package_ref[:8])
print(*p.holds, sep="\n")      # open decisions that block a run
```

The notebook `notebooks/01_agroforestry.ipynb` has one parameters cell (AOI, grid, decisions, simulation
settings, data source). It builds the run configuration, prints what is on hold, loads the layers (Earth Engine, or
synthetic data to test the pipeline), computes CIAT's partial terrain-and-soil index, runs the ordinal simulation,
compares both on maps and saves everything with a manifest to `runs/<fingerprint>/`.

Earth Engine: `pip install -e ".[gee,viz]"` and authenticate once. The notebook works on a 250 m grid, has an
optional quick check (section 3a) that pulls a small box and maps each layer, and saves every extracted layer in
`cache/layers/` so Earth Engine is called once. See `docs/DESIGN.md` for why 250 m.

`notebooks/02_variable_audit.ipynb` looks at the variables of an area one by one, in four groups (Climate variables, Soil, Terrain and
land, Project risk): a map, a histogram, the range, the share with no data, and a CHECK line when something looks wrong. It reads the
Hugging Face dataset (a local copy first, then only the window over the area) and Earth Engine. Install what it needs with
`pip install -e ".[audit]"`; the notebook tells you if something is missing in the Python it runs in.


<a id="how-it-connects-to-ciat"></a>

## Scoring package

The spatial notebook reads the seven verified files selected by `upstream/PIN.json`.
The pin records the publisher repository, exact commit, package version and file
hashes. Check the selected package offline with `python scripts/sync_upstream.py --check`.
To inspect another publisher, use a separate local store. [Publisher and update instructions](docs/DESIGN.md#updating-the-scoring-package)

<a id="rules-from-the-ciat-handoff"></a>

## Scoring rules

- Make corrections in the source package and refresh the pin after review.
- Unknown is never zero. Masks carry no weight. No per-pixel weight renormalisation.
- Operational factors (roads, market, tenure) stay outside the suitability score.
- Record the package pin, function fingerprints and decision references with every run.
