# Explainers & Metrics (Implementation Notes)

This file is a code-oriented reference for:
1) how each explainer produces an attribution vector, and
2) how each evaluator turns explanations into scalar metrics,
including which direction is preferred (higher-is-better vs lower-is-better) and why.

All file references below are relative to the repo root (e.g., `src/evaluators/...`).

## Common explanation schema (what evaluators consume)

All explainers return a per-instance dictionary shaped like the output of
`BaseExplainer._standardize_explanation_output()` (`src/explainers/base.py`):

```python
{
  "method": "...",
  "prediction": ...,
  "prediction_proba": ...,
  "attributions": [...],
  "feature_names": [...],
  "metadata": {...},
  "generation_time": ...,
  "instance": [...],
}
```

Most evaluators look for the attribution vector under one of:
`feature_importance`, `feature_importances`, `attributions`, or `importance`
in either the explanation root or `explanation["metadata"]`.

When an evaluator needs a baseline vector (for masking), it first tries
`explanation["metadata"]["baseline_instance"]` and otherwise falls back to a
constant baseline (often 0.0).

When an evaluator needs a scalar prediction value to compare, it generally prefers:
- binary classification: `prediction_proba[1]` (positive class probability),
- multiclass: `max(prediction_proba)`,
- otherwise: `prediction`.

## Explainers (how the attribution vectors are produced)

### LIME (`src/explainers/lime_explainer.py`)

Perturbs the instance with Gaussian noise (scaled by feature std), weights samples
by distance to the original instance, fits a Ridge regression surrogate, and uses
the surrogate coefficients as importances.

Key computation:
```python
perturbations = instance + rng.normal(0, std * noise_scale, size=(n_samples, n_features))
weights = np.exp(-(distances ** 2) / (kernel_width ** 2 + 1e-12))
importance = np.abs(Ridge(alpha=alpha).fit(perturbations, target, sample_weight=weights).coef_)
```

Directionality note: LIME returns non-negative importances (`abs(coef_)`), so the
sign information is intentionally dropped for this implementation.

### SHAP (`src/explainers/shap_explainer.py`)

- Tree models: uses `shap.TreeExplainer(model)` (fast, exact for many tree ensembles).
- Other models: uses `shap.KernelExplainer(predict_fn, background)` where `background`
  is a random subset of training rows (`background_sample_size`).
- If `shap` is not installed: falls back to a simple “replace-one-feature-with-mean”
  permutation-style score per feature.

Fallback sketch:
```python
perturbed = inst.copy()
perturbed[j] = bg_mean[j]
importances[j] = abs(base_pred - predict(perturbed))
```

### Integrated Gradients (`src/explainers/integrated_gradients_explainer.py`)

Uses a finite-difference approximation of gradients along the straight-line path from
a baseline to the instance, then integrates by averaging the gradients across steps.

Key computation:
```python
diff = instance - baseline
avg_grad = mean_i finite_difference_gradient(baseline + alpha_i * diff)
attributions = diff * avg_grad
```

Baseline is `train_mean` when available, else all-zeros.

### Causal SHAP (`src/explainers/causal_shap_explainer.py`)

Builds a simple causal parent graph using feature correlation (parents are earlier
features whose correlation exceeds `causal_shap_corr_threshold`). For each feature,
it samples coalitions that preferentially include parents, estimates the marginal
effect of adding the feature, and then rescales contributions so they sum to the
total effect (instance prediction minus baseline prediction).

### Prototype & Counterfactual (`src/explainers/example_based_explainer.py`)

Both are nearest-neighbour example-based explainers:
- `PrototypeExplainer`: nearest training example with the same predicted label.
- `CounterfactualExplainer`: nearest training example with a different label.

Attributions are per-feature absolute differences:
```python
attributions = np.abs(instance - reference_instance)
```

## Metrics / Evaluators (how scores are computed + which direction is better)

### Correctness / Deletion Check (`src/evaluators/correctness.py`)

Metric key: `correctness` (higher is better).

Intuition: if the explanation correctly identifies the important features, then
masking/removing those features should change the model output substantially.

Key computation:
```python
top = argsort(-abs(attributions))[:k]
perturbed = instance.copy()
perturbed[top] = baseline[top]
score = clip(abs(orig_pred - new_pred) / (abs(orig_pred) + 1e-8), 0, 1)
```

### Completeness (`src/evaluators/completeness.py`)

Metric keys:
- `completeness_drop` (higher is better): impact of deleting the “support”.
- `completeness_random_drop` (lower is better): average impact of deleting random features of the same size.
- `completeness_score` (higher is better): advantage over random deletion.

Support selection (tabular deletion mode):
```python
support = where(abs(attributions) >= magnitude_threshold)
if len(support) < min_features: support = argsort(-abs(attributions))[:min_features]
```

Deletion drop (same normalisation as correctness):
```python
drop = clip(abs(orig_pred - pred(mask(support))) / (abs(orig_pred) + 1e-8), 0, 1)
random_mean = mean_{trial}( clip(abs(orig_pred - pred(mask(random_support))) / denom, 0, 1) )
score = max(0, drop - random_mean)
```

Fast path (used when the explainer provides a “baseline prediction” scalar such as
LIME’s `baseline_prediction` or SHAP’s `expected_value`): compares `sum(attributions)`
to `(prediction - baseline_prediction)` and returns a [0, 1] agreement score. When
that baseline scalar is missing, the evaluator falls back to the deletion logic above.

### Continuity / Stability under small input noise (`src/evaluators/continuity.py`)

Metric key: `continuity_stability` (higher is better).

Intuition: small changes in the input should not radically change the explanation.

Key computation (tabular):
```python
noise = rng.normal(0, feature_std * noise_scale)
x_prime = x + noise
e_prime = explainer.explain_instance(x_prime)["attributions"]
score = abs(pearson_corr(e, e_prime))
```

### Relative Input Stability (RIS) (`src/evaluators/relative_stability.py`)

Metric key: `relative_input_stability` (lower is better).

Intuition: explanations should not change faster than the inputs.

Per perturbation:
```python
rel_attr = || (e - e_prime) / (abs(e) + eps_min) ||
rel_inp  = || (x - x_prime) / (abs(x) + eps_min) ||
ratio = rel_attr / max(rel_inp, eps_min)
```

The evaluator reports `max(ratio)` across `num_samples` perturbations.

### Consistency (`src/evaluators/consistency.py`)

Metric key: `consistency` (higher is better).

Intuition: if two instances receive the *same* explanation “pattern”, they should
also share the same predicted class.

Default discretisation hashes the sign pattern of the top-`n` (by magnitude) features:
```python
top = argsort(-abs(e))[:n]
token = hash(stack([top, sign(e[top])]).tobytes())
```

For each instance `i`, the score is:
`(# of other instances with same token and same predicted class) / (# with same token)`,
and the evaluator averages over instances.

### Non-sensitivity (`src/evaluators/non_sensitivity.py`)

Metric keys:
- `non_sensitivity_violation_fraction` (lower is better)
- `non_sensitivity_safe_fraction` (higher is better)
- `non_sensitivity_delta_mean` (lower is better)
- `non_sensitivity_zero_features` (count; no single “better” direction)

Intuition: features that receive (near) zero attribution should not matter; if you
swap them to a baseline value and the prediction changes, that’s a violation.

Key computation:
```python
zero = where(abs(attributions) <= zero_threshold)
for group in chunks(zero, features_per_step):
  x_prime = x.copy(); x_prime[group] = baseline[group]
  delta = abs(orig_pred - pred(x_prime))
  violations += len(group) if delta > delta_tolerance else 0
  safe += len(group) if delta <= delta_tolerance else 0
violation_fraction = violations / (violations + safe)
safe_fraction = safe / (violations + safe)
delta_mean = mean(deltas)
zero_features = len(zero)
```

Directionality note: `non_sensitivity_zero_features` is a diagnostic count (how many
features were considered “zero attribution”), not a pure quality score.

### Infidelity (`src/evaluators/infidelity.py`)

Metric key: `infidelity` (lower is better).

Intuition: an attribution vector should predict how the model output changes under
random feature perturbations.

Key computation:
```python
chosen = random_subset(k)
x_prime[chosen] = baseline[chosen] (+ optional noise)
delta_x = x - x_prime
approx_change = dot(attributions, delta_x)
true_change = pred(x) - pred(x_prime)
error = (approx_change - true_change) ** 2
infidelity = mean(error over samples)
```

### Monotonicity correlation (`src/evaluators/monotonicity.py`)

Metric key: `monotonicity` (higher is better; range [-1, 1]).

Intuition: larger attributions should correspond to larger model sensitivity.

The evaluator:
1) prepares attributions (optionally absolute value + max-normalisation),
2) sorts features by attribution,
3) perturbs features in small steps and measures mean squared prediction deltas,
4) returns a Spearman correlation between cumulative attribution mass and these
   variance terms.

Key computation sketch:
```python
sorted_idx = argsort(attrs)
for step_indices in chunks(sorted_idx, features_in_step):
  preds = [pred(mask(step_indices)) for _ in range(nr_samples)]
  variance_term = mean((preds - orig_pred)**2) * inv_pred_weight
  att_sum = sum(attrs[step_indices])
score = spearman(att_sums, variance_terms)
```

### Contrastivity (`src/evaluators/contrastivity.py`)

Metric keys:
- `contrastivity` (higher is better)
- `contrastivity_pairs` (count of evaluated off-class pairs)

Intuition: explanations should differ across predicted classes (target sensitivity).
For sampled off-class pairs, it computes similarity between attribution vectors
(default: SSIM-like function) and inverts it:
```python
score = 1 - similarity(e_i, e_j)
```

### Covariate Complexity / Regularity (`src/evaluators/covariate_complexity.py`)

Metric keys:
- `covariate_complexity` (lower is better)
- `covariate_regularity` (higher is better; defined as `1 - covariate_complexity`)

Intuition: diffuse attributions are harder to interpret; entropy captures “spread”.

Key computation:
```python
p = abs(e) / sum(abs(e))
entropy = -sum(p * log2(p))
normalized = clip(entropy / log2(n_features), 0, 1)
regularity = 1 - normalized
```

### Compactness / Size metrics (`src/evaluators/compactness.py`)

Metric keys (all higher is better):
- `compactness_sparsity`: fraction of features with `abs(e_i) <= zero_tolerance`
- `compactness_top5_coverage`: fraction of total mass in top-5 features
- `compactness_top10_coverage`: fraction of total mass in top-10 features
- `compactness_effective_features`: inverse participation ratio, normalised so 1.0 means “one dominant feature”

Key computation (tabular):
```python
imp = abs(e)
sparsity = 1 - count(imp > tol) / n_features
topk = sum(sort(imp, desc)[:k]) / sum(imp)
effective = 1 - ((1/sum((imp/sum(imp))**2)) - 1) / (n_features - 1)
```

### Confidence (`src/evaluators/confidence.py`)

Metric key: `confidence` (higher is better).

Intuition: repeated re-runs of a (stochastic) explainer should yield similar
attributions; wide variability implies low confidence.

How samples are collected:
- clone the explainer `n_resamples - 1` times with different random seeds,
- for IG optionally randomise the baseline (sample a training point),
- for SHAP/IG/causal-SHAP optionally add small input noise,
- stack attribution vectors into `samples` with shape `(n_resamples, n_features)`.

Per-feature confidence:
```python
width = percentile_upper(samples) - percentile_lower(samples)  # CI width
mean_abs = mean(abs(samples))
per_feature_conf = 1 - width / (mean_abs + width + 1e-8)
```

Aggregate confidence is a mean of `per_feature_conf` weighted by `mean_abs`.
The evaluator also stores `metadata["confidence_per_feature"]` for transparency.

## Metric direction used in Pareto encoding

`encode_pareto_fronts.py` negates metrics that are “lower is better” before doing
within-instance z-normalisation (so the model always sees “higher is better”).
See `NEGATE_METRICS` in `encode_pareto_fronts.py`.
