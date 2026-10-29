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
config/runs/     one TOML per run: target, soil choice, masks, decision references
src/nbs_runtime/ contract.py responses.py gates.py aggregate.py plan.py
scripts/         sync_upstream.py
tests/           conformance, pin integrity, policy, plan
notebooks/       thin notebooks per NbS
docs/
```

Planned, in this order: `data/` (SRTM slope, SoilGrids 0-30 cm, WorldCover, WDPA via GEE;
HF/local cache), `run.py` (first-pass M1), `manifest.py`, `outputs.py`, then M2 to M5.

## Open decisions (owner)

- Haiti applicability of the India pools and the exact AOI boundary (method lead, Diego).
- Adopt or replace the SoilGrids 0-30 cm proposal; record a decision reference (method lead).
- Assessed legal mask; until then the policy returns "unknown" for every pixel.
- Host inputs, GEE adapters, band and unit contracts (Diego, Brayden).
