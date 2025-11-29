# Configuration Schema

This directory contains registry-style YAML files that power the orchestrator. The
registries expose metadata about datasets, models, explainers, metrics, and experiments
so that we can enforce cross-cutting guarantees (e.g., only run tabular explainers on
tabular datasets) and eventually scale out to image/text modalities without changing
public APIs.

## Dataset registry (`dataset.yml`)

Each dataset entry lives under `datasets:` and inherits defaults via anchors. Required
fields:

- `type`: logical data type (`tabular`, `image`, `text`, …).
- `source`: human-readable origin (file path, URL, generator).
- `mandatory`: `true`/`false` flag used by tooling to highlight CI-critical assets.
- `description`: short summary for logs/CLIs.
- `loader`: module/factory pair that produces raw data.
- `loader_params`: keyword arguments forwarded to the loader.
- `split` (optional): override train/test splits for the adapter.
- `validation` (optional): structured metadata with `notes` and `overrides` for validator thresholds.

Example for adding a new tabular dataset that reuses the shared template:

```yaml
datasets:
  my_custom_dataset:
    <<: *tabular_dataset  # defined in the templates block
    source: local:data/my_dataset.parquet
    mandatory: true
    description: "Binary classification benchmark used in FooPaper2024."
    loader:
      module: src.datasets.tabular
      factory: TabularDataset.from_arrays
    loader_params:
      source: data/my_dataset.parquet
      target_column: label
    split:
      test_size: 0.2
      random_state: 123
    validation:
      notes: "Target column already encoded; no stratification needed."
```

Once the entry exists, experiments only need to reference `dataset: {key: my_custom_dataset}`
and the registry layer will enforce that only compatible models/explainers are paired with it.

## Model/explainer registries (`models.yml` / `explainers.yml`)

Model and explainer entries still carry their `module`/`class` (or `type`) definitions, but they
now inherit a `supported_data_types` list:

```yaml
templates:
  tabular_model: &tabular_model
    supported_data_types: [tabular]

random_forest:
  <<: *tabular_model
  module: sklearn.ensemble
  class: RandomForestClassifier
```

Keep this list in sync with the underlying implementation. When we add an image model, we only
need to set `supported_data_types: [image]` and the same `instantiate_model()` API will continue
to work across modalities.

Explainers use the same pattern. Combined with datasets having a `type`, this allows the shared
validation helper to prevent incompatible experiment tuples before any heavy computation runs.

## Experiments (`experiments.yml`)

Experiments reference registry keys explicitly:

```yaml
tabular_demo_suite:
  dataset: {key: tabular_toy}
  model: {key: random_forest}
  explainers:
    - {key: shap_default}
    - {key: lime_default}
```

The orchestrator flattens these references, asks each registry for the associated metadata, and
performs compatibility checks (dataset type vs. `supported_data_types`) before instantiating
loaders or estimators. Future modality-specific explainers/metrics can plug into the same guard
rails without introducing new user-facing configuration knobs.

## Validation thresholds (`validation.yml`)

`validation.yml` configures modality-level defaults for dataset quality checks (minimum sample
counts, class-imbalance tolerances, correlation/outlier/skewness limits, etc.). The
`TabularDataValidator` merges these thresholds with any per-dataset overrides from `dataset.yml`
before experiments run. Validation errors halt the experiment early; warnings are logged so
future explainers/metrics can reuse the same guardrails by data type.

## Hyperparameter tuning (`hyperparameters.yml`)

The benchmarking-ready tuner expects a dedicated config describing:

- `settings`: global knobs (CV folds, scoring metric, parallelism, optimization method).
- `grids`: parameter grids per model key. Only models that exist in both repos (e.g.,
  `random_forest`, `gradient_boosting`, `mlp_classifier`) currently include grids; you can add
  more entries later using the same shape.

When `run_experiment(..., tune_models=True)` is invoked (via the CLI flags or programmatically),
the orchestrator loads these grids, runs scikit-learn's `GridSearchCV`, persists the best params
under `saved_models/tuning_results/<dataset>/<model>.json`, and reuses them when
`use_tuned_params=True`.
