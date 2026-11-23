# CARE-IPS Architecture

The package follows a layered structure: data -> agents -> evidence -> consensus ->
reasoning -> governance -> feedback -> evaluation.

The `scripts/` directory contains entry points only. Reusable scientific logic belongs
under `src/care_ips/`. Every run should write to a unique `results/runs/<run_id>/`
directory with configuration, metrics, predictions, and logs.

## Agent heterogeneity
Feature views are explicitly assigned to agents. Distinct names do not prove
independence: report feature overlap, model similarity, provenance similarity,
and historical error correlation. Agent A/B must not be treated as independent
if their feature sets and training procedures are effectively identical.
