# Human Centered XAI

Human Centered XAI (HC-XAI) is a framework for generating, evaluating, and personalizing explanations of model predictions. It surfaces multiple explanation methods for each instance, lets people pick the ones they find most useful, and ultimately aims to learn a latent explanation ranker that can proactively recommend explanations aligned with each individual’s preferences.

## Project Overview

1. Train or load a predictive model and run a suite of explainers on tabular datasets.  
2. Present the resulting local explanations to human participants so they can label or rank their preferred styles.  
3. Use those interactions to fit a ranker that maps context (model, instance, user) to the explanation style most likely to satisfy that person.

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

### 2. OUTPUT-COMPLETENESS

### 3. CONSISTENCY

### 4. CONTINUITY
Continuity currently reuses the stability-for-slight-variations test (Co-12 Section 6.4). For a handful of instances we apply a tiny Gaussian perturbation to the feature vector (noise is scaled by training-set standard deviation when available), re-run the explainer on the perturbed instance, and compute the absolute correlation between the perturbed and original attribution vectors. Averaging those correlations yields the `continuity_stability` score—values close to 1 indicate explanations remain smooth under small input changes. Config knobs: `max_instances` (how many explanations to perturb), `noise_scale` (perturbation magnitude), and `random_state`.

### 5. CONTRASTIVITY

### 6. COVARIATE COMPLEXITY
Covariate complexity from Co-12 Section 6.6 is implemented by `CovariateComplexityEvaluator` (`src/evaluators/covariate_complexity.py`). It iterates over each instance’s attribution vector, converts magnitudes into a probability distribution, and reports the normalized Shannon entropy (`covariate_complexity`) plus its complement (`covariate_regularity`). Enable it through the `covariate_complexity` entry in `src/configs/metrics.yml`; all bundled experiment suites request it so the new scores automatically appear next to correctness, continuity, and compactness.

### 7 COMPACTNESS
`CompactnessEvaluator` (`src/evaluators/compactness.py`) aggregates the Size-style metrics from Co-12 Section 6.7. For each explanation it measures (a) sparsity—the fraction of near-zero attributions, (b) top-5 / top-10 coverage—how much attribution mass lives on the most important features, and (c) effective feature count—an inverse participation ratio normalized so 1.0 means a single dominant feature. The evaluator averages those scores across instances and is wired up through the `compactness_size` entry in `src/configs/metrics.yml`; every experiment suite requests it so you can monitor explanation brevity alongside correctness and regularity.

### 8 COMPOSITION

### 9 CONFIDENCE

### 10 CONTEXT

### 11 COHERENCE

### 12 CONTROLLABILITY
