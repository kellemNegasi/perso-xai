# Human Centered XAI

Human Centered XAI (HC-XAI) is a framework for generating, evaluating, and personalizing explanations of model predictions. It surfaces multiple explanation methods for each instance, lets people pick the ones they find most useful, and ultimately aims to learn a latent explanation ranker that can proactively recommend explanations aligned with each individual’s preferences.

## Project Overview

1. Train or load a predictive model and run a suite of explainers on tabular datasets.  
2. Present the resulting local explanations to human participants so they can label or rank their preferred styles.  
3. Use those interactions to fit a ranker that maps context (model, instance, user) to the explanation style most likely to satisfy that person.

### Running Experiments

The old exploratory notebooks now delegate to a reusable orchestrator so that scripted runs and notebooks share the same code path. To reproduce the JSON artifacts shown in `notebooks/notebooks/results_per_instance_*` simply call:

```python
from pathlib import Path
from src.orchestrators.metrics_runner import run_experiment

run_experiment(
    "breast_cancer_lr_suite",
    max_instances=10,
    output_path=Path("notebooks/notebooks/results_per_instance_breast_cancer_lr_suite.json"),
)
```

The runner pulls the dataset/model/explainer/metric configs from `src/configs/*.yml`, instantiates everything, and attaches both per-instance and batch metrics using the capability metadata each evaluator now exposes.

## Current Explainers

### SHAP (`shap_default`)
Hybrid Tree/Kernel SHAP implementation [(`src/explainers/shap_explainer.py`)](src/explainers/shap_explainer.py). Tree models use `shap.TreeExplainer`; other models fall back to Kernel SHAP with a randomly sampled background set. Parameter: `background_sample_size` (default 100) controls how many training points form the background distribution for Kernel SHAP; larger values reduce variance but increase runtime.

### LIME (`lime_default`)
Local linear surrogate around each instance [(`src/explainers/lime_explainer.py`)](src/explainers/lime_explainer.py). Parameters: `lime_num_samples` (number of noisy perturbations drawn per instance), `lime_noise_scale` (standard deviation multiplier for Gaussian perturbations relative to feature std), and `lime_kernel_width` (RBF width that weights perturbations by similarity). Higher sample counts and smaller noise yield smoother but more expensive explanations.

### Integrated Gradients (`integrated_gradients_default`)
Finite-difference approximation of Integrated Gradients for tabular models [(src/explainers/integrated_gradients_explainer.py)](src/explainers/integrated_gradients_explainer.py). Parameters: `ig_steps` (number of interpolation points between baseline and instance) sets the Riemann approximation resolution, and `ig_epsilon` (finite-difference step) controls gradient accuracy. Baseline defaults to the training mean when available, otherwise zeros.

### Causal SHAP (`causal_shap_default`)
Custom correlation-aware SHAP variant that infers a simple causal ordering from feature correlations before sampling coalitions [(`src/explainers/causal_shap_explainer.py`)](src/explainers/causal_shap_explainer.py). Parameters: `causal_shap_corr_threshold` determines when two features are considered linked (higher threshold → sparser graph) and `causal_shap_coalitions` sets the number of Monte Carlo coalitions sampled per feature (more samples reduce variance but cost more model calls).

## Implemented Metrics

### 1. Correctness (Deletion Check)
The current correctness metric follows the deletion check from “From Anecdotal Evidence to Quantitative Evaluation Methods”: for each explanation we rank features by absolute attribution, remove the top-k according to the evaluator config, and re-run the model. The score is the normalized drop in the model’s scalar output (probability when available), averaged over all evaluated instances. When explainers provide a `baseline_instance` in their metadata we use those feature values for masking; otherwise the evaluator falls back to a constant baseline (default 0.0). This exposes how strongly the highlighted features control the prediction and lets the same logic power both the `correctness` and `output_completeness_deletion` metrics.

`NonSensitivityEvaluator` (`src/evaluators/non_sensitivity.py`) complements the deletion check by perturbing every feature whose attribution magnitude falls below a configurable `zero_threshold`. If swapping those “zero-importance” features with the explainer baseline alters the prediction more than a tolerance, the evaluator counts a violation. It reports the average violation fraction, the safe (non-violating) fraction, how many zero-attribution features were tested, and the mean prediction delta via the `non_sensitivity_*` metric keys so you can monitor whether sparse explanations are hiding important contributors. This implementation adapts the Quantus NonSensitivity metric (https://github.com/understandable-machine-intelligence-lab/Quantus) to the HC-XAI orchestration stack.

### Monotonicity Correlation
`MonotonicityEvaluator` (`src/evaluators/monotonicity.py`) adapts the Quantus `MonotonicityCorrelation` metric from Nguyen & Rodríguez Martínez (2020). It sorts features by (optionally normalised) attribution magnitude, perturbs them in ranked batches, and measures the Spearman correlation between cumulative attribution mass and the squared prediction deltas produced by those perturbations. Scores near +1 indicate that large attributions always precede proportionally large model changes (faithful ordering); scores near −1 reveal inverted rankings. Configuration knobs: `nr_samples` (Monte Carlo perturbations per step), `features_in_step`, `noise_scale` for optional stochastic replacements, and `default_baseline` when explainers omit `baseline_instance`.

### Infidelity
`InfidelityEvaluator` (`src/evaluators/infidelity.py`) implements the Yeh et al. (2019) infidelity loss. For each instance it samples random feature subsets, replaces them with an explainer-provided baseline, and compares two quantities: the dot product between the attribution vector and the input delta versus the actual prediction change induced by that perturbation. The mean squared difference is the score (`infidelity`), so lower is better—0 means the explanation perfectly predicts the model’s response under those perturbations. Tunables: `n_perturb_samples`, `features_per_sample`, `noise_scale` for stochastic baselines, and `default_baseline`.

### 2. Completeness (Deletion Advantage)
`CompletenessEvaluator` (`src/evaluators/completeness.py`) implements the preservation/deletion check from the Co-12 paper. For each explanation it masks **all** features whose attribution magnitude exceeds the configured threshold, re-evaluates the model, and measures how much the prediction changes relative to the original output (`completeness_drop`). It then samples equally sized random feature sets, masks them, and reports the average drop from those baselines (`completeness_random_drop`). The final `completeness_score` is the clipped advantage (`max(0, drop - random_drop)`), so explanations only receive credit when deleting their highlighted features disrupts the model substantially more than random noise. Wire it into any experiment suite by adding `completeness_deletion` to the `metrics` list (see `src/configs/experiments.yml`); default parameters live in `src/configs/metrics.yml`.

### 3. CONSISTENCY
`ConsistencyEvaluator` (`src/evaluators/consistency.py`) adapts the Quantus consistency test (https://github.com/understandable-machine-intelligence-lab/Quantus). It discretises each attribution vector (default: the sign of the first `n` components) so explanations that highlight the same pattern share an identifier, then measures the fraction of same-identifier pairs whose predicted class matches. High scores mean the explainer only reuses identical narratives on instances where the model actually makes the same decision. Configure it by adding `consistency_local` in `metrics.yml` and optionally swapping `discretise_func`/`discretise_kwargs` to change how explanations are bucketed.

### 4. CONTINUITY
Continuity reuses the stability-for-slight-variations test (Co-12 Section 6.4). For each explanation we sample a tiny Gaussian perturbation (noise is scaled by training-set standard deviation when available), re-run the explainer on the perturbed instance, and compute the absolute correlation between the perturbed and original attribution vectors. Scores live on \[0, 1]; values near 1 mean the explanation barely changes under small input noise. The evaluator now supports both per-instance reporting (when `current_index` is provided) and batch-level summaries. Config knobs: `max_instances` (how many explanations to perturb when aggregating), `noise_scale` (perturbation magnitude), and `random_state`.

### Relative Input Stability
`RelativeInputStabilityEvaluator` (`src/evaluators/relative_stability.py`) adapts the RIS metric from Agarwal et al. / Quantus (https://github.com/understandable-machine-intelligence-lab/Quantus). It perturbs each instance multiple times, reruns the explainer, and reports the maximum ratio between the relative attribution change and the relative input change. Values close to zero indicate explanations that move no faster than their inputs; large values flag unstable attribution surfaces. Enable it through `relative_input_stability` in `metrics.yml` and adjust `num_samples`, `noise_scale`, or `eps_min` to control the stress level.

### 5. CONTRASTIVITY
`ContrastivityEvaluator` (`src/evaluators/contrastivity.py`) adapts the Random Logit / target-sensitivity metric popularised by Sixt et al. (2020) and the Quantus library: for each explanation it randomly samples off-class references, measures Structural Similarity (SSIM) between attribution vectors, and reports `1 - SSIM` so higher values mean stronger disagreement across classes. The evaluator supports per-instance use (anchoring the comparisons on the requested explanation) as well as batch aggregates. Parameters: `pairs_per_instance` controls how many off-class references to sample per explanation, `similarity_func` allows swapping the SSIM variant, and `normalise` toggles L1 normalisation before similarity is computed.

### 6. COVARIATE COMPLEXITY
Covariate complexity from Co-12 Section 6.6 is implemented by `CovariateComplexityEvaluator` (`src/evaluators/covariate_complexity.py`). It iterates over each instance’s attribution vector, converts magnitudes into a probability distribution, and reports the normalized Shannon entropy (`covariate_complexity`) plus its complement (`covariate_regularity`). Enable it through the `covariate_complexity` entry in `src/configs/metrics.yml`; all bundled experiment suites request it so the new scores automatically appear next to correctness, continuity, and compactness.

### 7 COMPACTNESS
`CompactnessEvaluator` (`src/evaluators/compactness.py`) aggregates the Size-style metrics from Co-12 paper. For each explanation it measures (a) sparsity—the fraction of near-zero attributions, (b) top-5 / top-10 coverage—how much attribution mass lives on the most important features, and (c) effective feature count—an inverse participation ratio normalized so 1.0 means a single dominant feature. The evaluator averages those scores across instances and is wired up through the `compactness_size` entry in `src/configs/metrics.yml`; every experiment suite requests it so you can monitor explanation brevity alongside correctness and regularity.

### 8 COMPOSITION
The Co-12 “composition” metric described in *From Anecdotal Evidence to Quantitative Evaluation Methods* only specifies scoring procedures for textual (e.g., rationale quality, linguistic coherence) or image-based (mask/segment agreement) explanations. Because our work currently focuses on attribution vectors for tabular data, we do not have an attribution-oriented adaptation of this metric yet. Future work should first surface a published definition (or propose one) for how composition ought to be measured for attribution methods before we can add an evaluator.

### 9 CONFIDENCE
`ConfidenceEvaluator` (`src/evaluators/confidence.py`) estimates how stable an explainer’s attributions are across repeated runs. For seed-dependent methods (LIME, Kernel/Causal SHAP) it clones the explainer, reruns it with different RNG seeds/background draws, and stacks the resulting attribution vectors; for deterministic methods (Integrated Gradients) it injects randomness via alternate training baselines and small input noise. The evaluator converts the empirical distribution into per-feature confidence scores using percentile widths, weights those by attribution magnitude, and reports the aggregate as a single `confidence` scalar (also attaching the feature-level confidences to the explanation metadata). Configure `confidence` in `src/configs/metrics.yml` to enable it, adjusting `n_resamples`, `ci_percentile`, or `noise_scale` if you want tighter or looser intervals.

### 10 CONTEXT
Context metrics from Co-12 focus on whether explanations respect user needs. We have not wired them up because they require either (a) user-specific action-cost metadata to run the “Pragmatism” test for counterfactuals (assign feature-change costs per person and score explanations by how feasible their suggestions are) or (b) simulated user studies like Ribeiro et al.’s trust/model-selection experiments that assume knowledge of which features users consider untrustworthy. Once we gather such user-context data, we can integrate those two approaches to quantify how well explanations honor stakeholder constraints without running full human studies.

### 11 COHERENCE
We have not yet implemented coherence scoring because it requires an annotated dataset that explicitly states which features should matter for each record. Once such labels exist the metric will mirror the Co-12 “alignment with domain knowledge” test by computing a rank-based similarity (e.g., Spearman/Kendall) between predicted feature-importance vectors and the annotations. If no ground-truth exists we will approximate coherence via “XAI methods agreement,” i.e., measure the same rank-based similarity against a trusted reference explainer and treat close alignment as evidence of coherence.

### 12 CONTROLLABILITY
Controllability evaluation from Co-12 requires interactive explanation methods and instrumentation to measure the impact of user feedback. We do not implement it yet because we lack experiments where humans can iteratively steer explanations or provide concept-level preferences. When we do, we will follow the survey’s “Human Feedback Impact” setup: quantify how explanation quality (e.g., textual accuracy, attribution agreement) improves after each feedback round and optionally report the concept-level satisfaction ratio (fraction of user-requested concepts present minus forbidden concepts included) as Chen et al. suggest.
