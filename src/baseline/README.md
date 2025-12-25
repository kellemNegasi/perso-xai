# AutoXAI Baseline (Cached) for HC-XAI

This baseline reproduces AutoXAI’s *selection + scalarization* idea, but runs it **on top of cached HC-XAI per-instance metrics** so we can compare recommendation rankings without rerunning the full HC-XAI explainer pipeline.

Implementation lives in `src/baseline/autoxai.py`.

## What It Does

- Loads `*_metrics.json` files under a HC-XAI run directory, e.g. `results/full_run_dec8/metrics_results/<dataset>/<model>/lime_metrics.json`.
- Builds candidates keyed by `(dataset_index, method_variant)` for the requested `--methods` (typically `lime` and `shap`).
- Computes an AutoXAI-style objective (by default for `--persona autoxai`):
  - `robustness`: minimize `relative_input_stability`
  - `fidelity`: minimize `infidelity`
  - `conciseness`: maximize `compactness_effective_features`
- Standardizes each term (Std or MinMax), applies weights, scalarizes, and ranks:
  - `best_variant_by_method` (best hyperparameter variant per method)
  - `method_ranking` (method-to-method comparison based on each method’s best variant)
- Optionally compares against HC-XAI **pairwise labels** (parquet files emitted by `candidates_pair_ranker.py`) via pairwise accuracy.

## Scaling (Important)

`--scaling-scope` controls *what population is used to fit the per-term scaler*:

- `trial` (default): AutoXAI-like “trial-history” scaling.
  - `--hpo grid`: scalers fit over **all** variants in the candidate pool (same as evaluating the full grid).
  - `--hpo random` / `--hpo gp`: scalers fit **sequentially** over the evaluated trial history so far (AutoXAI-like on-the-fly scaling).
- `global`: scalers fit over all `(instance,variant)` candidate values (older behaviour).
- `instance`: scalers fit per instance across its variants.

## Run One Experiment

From repo root:

```bash
.venv/bin/python -m src.baseline.autoxai \
  --results-root results/full_run_dec8 \
  --dataset open_compas \
  --model mlp_classifier \
  --methods lime shap \
  --persona autoxai \
  --hpo grid \
  --scaling Std \
  --scaling-scope trial \
  --output results/full_run_dec8/baslines/autoxai_baseline__open_compas__mlp_classifier__autoxai.json \
  --require-write
```

If we omit `--output`, the module defaults to writing:
- `<results-root>/autoxai_baseline__<dataset>__<model>__<persona>.json`

It always prints the JSON report to stdout; `--require-write` makes it error if it cannot write the file.

## Personas / Objectives

`--persona` selects a preset objective:

- `autoxai`: robustness/fidelity/conciseness (AutoXAI-like mapping to HC-XAI cached metrics)
- `layperson`: compactness/contrastivity/stability (matches HC-XAI pair-label priorities)
- `regulator`: faithfulness/completeness/consistency/compactness (matches HC-XAI pair-label priorities)

You can override presets with `--objective name:direction:metric_key[:weight] ...`.

## Run All Datasets/Models

Use the runner script:

- `src/baseline/run_autoxai_all.sh`

Example:

```bash
src/baseline/run_autoxai_all.sh --results-root results/full_run_dec8 --persona autoxai --hpo grid --scaling-scope trial
```

By default it writes reports to:
- `<results-root>/baslines/`

and logs to:
- `<output-dir>/logs/`

Use `--dry-run` to preview what would run:

```bash
src/baseline/run_autoxai_all.sh --dry-run --results-root results/full_run_dec8 --persona autoxai 2>&1 | head
```

## Comparing to HC-XAI Pair Labels

If you have pair-label parquet files for a persona, you can pass them directly:

- `--pair-labels results/full_run_dec8/candidate_pair_rankings_layperson/<dataset>__<model>_pareto_pair_labels.parquet`
- `--pair-labels results/full_run_dec8/candidate_pair_rankings_regulator/<dataset>__<model>_pareto_pair_labels.parquet`

The all-runs script auto-adds `--pair-labels` for `layperson`/`regulator` when the file exists.

