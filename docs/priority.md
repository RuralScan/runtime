# T2/T5 classification and user weighting

This is a reference evaluator for aligned pixel sequences. It implements the packaged, SHA-pinned
RuralScan/method `priority-contract.json`. It is separate from the existing T4 suitability notebook;
it does not fetch spatial datasets, choose thresholds, resolve a CRS or approve scientific rules.

## Run the complete illustration

```sh
uv sync --locked
uv run python -m nbs_runtime.priority config/priority/illustration.json --output /tmp/priority-example.json
uv run pytest -q tests/test_priority.py
uv run python scripts/assess_critic.py
```

The illustration has two concerns: poverty and flood, weighted 3:1. The first two cells score
0.8325 and 0.745; one incomplete cell and one unknown mask remain unknown; the outside cell is excluded.
These thresholds and values are hypothetical, not calibrated defaults. Changing the weights to 1:3
reverses the first two cells’ ranking.

## Input and output

`config` declares `format`, `purpose`, context, variables, rules, representative selection and weights.
`layers` maps qualified IDs such as `T2:flood_example` to `values`, exact `unit` and `source_ref`.
`opportunity` is an aligned list of true, false or null. Optional `references` maps classification IDs
to reference samples for terciles. The example is a complete input template.

Variables are the normalized runtime projection of T2/T5 definitions: table, variable_id, unit,
mcda_role, concern_id and classification_id. Optional VONT variable, dataset, hazard, scenario and
theme metadata can be retained. Dataset bindings and validity checks must be resolved by the caller.
All selected positive-weight rules are checked before scoring. Unselected definitions are retained
in the input but are not certified by this execution.

Threshold rules supply three finite, ascending upper-inclusive cutoffs. Class labels reverse for
`lower_is_more_concern`. Terciles use an explicit reference, preserving ties; they need a separate
none condition to provide a zero class. Null, NaN and infinite observations become unknown. Other
out-of-domain values must be masked by the data adapter. No implicit unit conversion is performed.

Only one chosen representative per shared concern enters the stack. Each needs a reason. This
enforces declared identities; it cannot discover incorrectly labelled duplicate concerns. Positive
user weights normalize once across the stack. Missing positive-weight inputs produce unknown scores;
zero-weight inputs are not required. Descriptors and investment-risk screens cannot enter the sum.

The output records context, method contract pin, configuration and mask hashes, data references and
hashes, resolved rules and reference hashes, selections, normalized weights, per-cell contributions,
status counts, and raw/classified Pearson diagnostics within the known opportunity space. Keep the
input bundle alongside it: hashes identify inputs but do not contain them. There is no second
min–max stretch, quintile classification, theme multiplier or T6 response multiplier.

## CRITIC assessment

`critic_weights` is a separate diagnostic API. Inputs must already share concern direction. Its
original variant uses min–max scaling; `fixed_0_1` explicitly compares scores without that stretch.
It uses a shared complete-case sample, reports constants and returns no weights when information is
zero. It never changes `run` weights. The tests reproduce the 1995 paper’s example within 0.001.
The assessment script demonstrates area, discretisation, scaling and duplicate sensitivity.

The method repository contains the scientific assessment at
`methodology-reference/reviews/critic-assessment-2026-11-24.md`.

## Before a pilot

Supply reviewed, context-specific rules and data bindings. An `analysis` run rejects explicitly
illustrative rule references, but a nonempty decision reference is not authenticated human approval.
Reviewers still approve science through method PRs. The caller must establish grid/CRS alignment,
valid-domain masks, time/scenario consistency and geographic support. Evaluate coverage and ranking
sensitivity. This addition does not claim a deployed pilot or change evidence registers.
