# nbs-ruralscan-runtime

Runtime for **NbS Rural Scan**. It takes the recipes produced by the evidence skill (one file per variable:
a suitability function and a weight) and turns them into suitability maps, with the uncertainty that comes
with them. The methodology is approach 3 of the RuralScan MCDA methodology:

**S = L · G · M**

- **L**: the lowest score among the *limiting* variables (no weights): the most limiting factor rules.
- **G**: 1 where the land cover is one where the NbS can be implemented, 0 where not.
- **M**: the weighted sum of the *modulating* scores, recipe weights renormalised among them.

A value outside the nodes of its recipe has no evidence behind it: the cell is *not evaluated*, never 0.

## Inputs

| What | Where | Who writes it |
|---|---|---|
| Recipes: function (`x`, `score`, linear) and `weight` per variable | `config/recipes/<nbs>/<family>/<variable>.json` | the evidence skill |
| Role of each variable: `limiting` or `modulating` | `config/roles.toml` (default, with per-NbS overrides) | runtime |
| Datasets: where each variable is read from, unit, scale | `src/nbs_runtime/variables.py` | runtime |
| A run: area, grid, eligible land cover, sensitivity | `config/runs/*.toml` (and the notebook's parameters cell) | user |

The runtime never changes a recipe. A recipe that does not validate, a variable with no role or no dataset,
or a unit that differs between the recipe and the dataset blocks the run (`plan()` lists it).

## Quickstart

```bash
uv sync                                                   # runtime and dev tools
uv run pytest                                             # tests
uv run ruff check .                                       # lint
uv run python scripts/demo_suitability.py --png demo.png  # the whole index on synthetic layers, no Earth Engine
```

Notebooks: select `.venv` as the kernel (VS Code), or `uv run --with jupyter jupyter lab`.

- `notebooks/01_suitability.ipynb` runs the whole workflow, one step per section: recipes and their curves,
  plan, grid, inputs, scores, L, G, M and S, one pixel explained step by step, classes, sensitivity
  (equal weights, Dirichlet weights, leave one variable out) and the outputs (GeoTIFFs, arrays and a manifest
  in `runs/<nbs>-<family>-<fingerprint>-<source>/`). Set `DATA_SOURCE = "synthetic"` to try it offline.
- `notebooks/02_variable_audit.ipynb` looks at every catalog variable of an area one by one (map, histogram,
  range, share with no data, a CHECK line when something looks wrong) and says where each is used in the recipes.

Earth Engine: `uv sync --extra gee --extra viz --extra geo` and authenticate once (the project id is read from
`EE_PROJECT` in `.env`). Every extracted layer is saved in `cache/layers/`, so Earth Engine is called once per
grid. `scripts/check_gee.py` loads every variable of a family on a tiny box first.

```python
from nbs_runtime.recipes import load_family
from nbs_runtime.suitability import compute, classify

fam = load_family("agroforestry", "cross_family")
result = compute(fam, layers, eligible)       # layers: {variable: array}, eligible: G (1/0/NaN)
classes = classify(result.S)
```

## Adding an NbS or a family

1. The skill writes `config/recipes/<nbs>/<family>/<variable>.json`.
2. Give every new variable a role in `config/roles.toml` (or an override under `[<nbs>]`).
3. Give every new variable a catalog entry in `variables.py` (dataset, unit, scale).
4. `uv run pytest tests/test_recipes.py tests/test_variables.py` checks all three.
5. Run the notebook with `NBS` and `FAMILY` set.

## Rules

- Unknown is never zero: no data, or a value outside the evidence, leaves a cell not evaluated.
- Limiting variables carry no weight; modulating weights are never renormalised per pixel.
- Operational factors (roads, market, tenure) and opportunity variables stay outside the suitability score.
- Every run records the recipe hashes, the roles, the weights used and the configuration (`manifest.json`).

Contact: Diego Agudelo, d.agudelo@cgiar.org

## Other readers (under review)

The two readers below come from earlier pull requests and are not used by the suitability workflow.

### Read a six-practice recipe release

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

### Accepted AI scientific effects

New T3/T6 scientific judgments are authored by AI and accepted in the
[RuralScan/evidence synthesis workflow](https://github.com/RuralScan/evidence/blob/main/workflows/synthesis/README.md).
Runtime does not infer direction, strength, confidence, context or weights. Use an
accepted-effects package with an independently trusted manifest hash:

```python
from pathlib import Path
from nbs_runtime.accepted_effects import AcceptedEffects

accepted = AcceptedEffects(Path("/path/to/accepted-effects"), manifest_sha256=TRUSTED_SHA256)
row = accepted.row("agroforestry", "T6", "accepted_record_id")
provenance = accepted.provenance("agroforestry", "T6", "accepted_record_id")
```

The reader checks package inventory, file hashes and exact accepted-row provenance.
It returns the selected record unchanged. Missing, withdrawn, held or stale records
cannot become runtime values, and there is no nearest-context or historical fallback.
Packages may contain only some cells: consumers must explicitly report missing accepted
outputs. Withdrawal provenance remains readable. Existing pinned recipe-store APIs
remain historical consumers; they do not confer AI scientific acceptance on old rows.
This reader supplies accepted scientific values to consumers; it does not add a new
geospatial calculation or perform fresh scientific synthesis.
