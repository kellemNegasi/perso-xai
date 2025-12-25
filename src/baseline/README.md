# AutoXAI Baseline (Cached) for HC-XAI

This baseline reproduces AutoXAI’s *selection + scalarization* idea, but runs it **on top of cached HC-XAI per-instance metrics** so we can compare recommendation rankings without rerunning the full HC-XAI explainer pipeline.

Entry point lives in `src/baseline/autoxai.py`, with implementation split across:
- `src/baseline/autoxai_runner.py` (pipeline)
- `src/baseline/autoxai_scoring.py` (scoring/scaling)
- `src/baseline/autoxai_hpo.py` (HPO schedules/selection)
- `src/baseline/autoxai_evaluation.py` (pair-label + top-k evaluation)
- `src/baseline/autoxai_objectives.py` / `src/baseline/autoxai_config.py` / `src/baseline/autoxai_data.py` / `src/baseline/autoxai_utils.py`

## What It Does

- Loads `*_metrics.json` files under a HC-XAI run directory, e.g. `results/full_run_dec8/metrics_results/<dataset>/<model>/lime_metrics.json`.
- Builds candidates keyed by `(dataset_index, method_variant)` for the requested `--methods` (e.g. `lime`, `shap`, `integrated_gradients`, `causal_shap`).
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

## Holdout Evaluation (Match HC-XAI Test Split)

To evaluate AutoXAI on the **same held-out instances** used by HC-XAI preference-learning, pass the split file:

- `--hc-xai-split-json results/full_run_dec8/preference_learning/<persona>/<dataset>__<model>_pareto/processed/splits.json`
- `--split-set test` (default) to restrict scoring to HC-XAI's test instances.

When `--pair-labels` is present, AutoXAI also emits an HC-XAI-style `top_k_evaluation` report:

- `--top-k 3 5` (default: `3 5`)

Convenience script (recommended):

```bash
src/baseline/run_autoxai_holdout_eval.sh \
  --results-root results/full_run_dec8 \
  --dataset open_compas \
  --model mlp_classifier \
  --persona layperson \
  --split-set test \
  --top-k "3 5"
```

To run the same holdout evaluation over **all** dataset/model pairs for a persona:

```bash
bash src/baseline/run_autoxai_holdout_eval_all.sh \
  --results-root results/full_run_dec8 \
  --persona layperson \
  --split-set test \
  --top-k "3 5" \
  --jobs 4
```
