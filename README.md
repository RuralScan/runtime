# nbs-ruralscan-runtime

Runtime for **NbS Rural Scan**. It reads the generated T4 developer package published by
[CIAT/nbs_ruralscan](https://github.com/CIAT/nbs_ruralscan) (methodology and evidence live
there) and turns it into spatial results. This repository holds no methodology.

Current scope: the approved **planted-silvoarable first pass** (agroforestry): slope 50%,
soil organic carbon 25%, pH 25%, plus a host gate and a legal-exclusion mask. It is a
partial terrain-and-soil index, not complete agroforestry suitability.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python scripts/sync_upstream.py --check     # pinned CIAT files are intact
pytest                                      # 40 response cases + 6 policy cases + checks
```

```python
from pathlib import Path
from nbs_runtime.plan import plan

p = plan(Path("config/runs/agroforestry_planted_first_pass.toml"))
print(p.package_version, p.package_ref[:8])
print(*p.holds, sep="\n")      # open decisions that block a run
```

## How it connects to CIAT

`upstream/` holds a read-only, hash-verified copy of one immutable package version
(about 0.7 MB). `upstream/PIN.json` records the CIAT commit and file hashes. To adopt a
newer package see [docs/DESIGN.md](docs/DESIGN.md#updating-from-ciat).

## Rules (from the CIAT handoff)

- Never edit `upstream/`; report contract gaps to CIAT instead of fixing values here.
- Unknown is never zero. Masks carry no weight. No per-pixel weight renormalisation.
- Operational factors (roads, market, tenure) stay outside the suitability score.
- Record the package pin, function fingerprints and decision references with every run.

Contact: Diego Agudelo, d.agudelo@cgiar.org
