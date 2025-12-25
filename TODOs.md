# TODOs: Persona Simulation Alternatives (Notes + References)

This document collects options for simulating user/persona preferences over explanation quality metrics (e.g., HC-XAI metrics) beyond a strict priority/lexicographic rule.

## 1) Weighted Utility (Additive Model)

**Idea**
- Represent a persona by a weight vector over metrics (or metric groups).
- Convert each candidate explanation method/variant into a feature vector of metric values.
- After normalizing metrics to comparable scales, compute a single utility score:
  - `U(method_variant | instance) = Σ_j w_j * z_j(metric_j)`
  - where `z_j` is a normalization/transform (e.g., z-score, min-max, log, sigmoid) and `w_j ≥ 0`.

**Why it’s more realistic than strict priorities**
- Allows tradeoffs (e.g., a user may accept slightly lower compactness if faithfulness improves a lot).
- Easy to interpret and to run sensitivity analyses (vary weights).

**How it’s used in research**
- Multi-attribute utility / value theory: additive utility under independence assumptions.
- Multi-criteria decision making and policy analysis often rely on weighted sums for tradeoff analysis.

**Practical notes for HC-XAI**
- Choose metric directions consistently (higher-better). Invert “loss-like” metrics (e.g., infidelity, instability).
- Prefer per-instance normalization if your “ground truth” is per-instance pairwise judgments.
- Report robustness to weights: a grid/random sweep of weights shows how stable conclusions are.

**References (To be verfied)**
- Keeney, R. L., & Raiffa, H. *Decisions with Multiple Objectives: Preferences and Value Tradeoffs.* (book; OpenLibrary work `https://openlibrary.org/works/OL4107460W`; example ISBN-10: `0521441854` / `0521438837`).
- Saaty, T. L. *The Analytic Hierarchy Process.* (1980; McGraw-Hill; OpenLibrary work `https://openlibrary.org/works/OL2996560W`; ISBN-10: `0070543712`).
- von Neumann, J., & Morgenstern, O. *Theory of Games and Economic Behavior.* (1944; book; typically cited as foundational utility theory; ISBN varies by edition).

## 2) Learned Preference Model from Pairwise Choices

**Idea**
- Instead of choosing weights by hand, fit a probabilistic preference model from pairwise comparisons.
- Training data: tuples `(instance, A, B, label)` where label indicates whether the user preferred A or B.
- Model produces `P(A ≻ B | features(A,B,instance))`.

**Common models**
- **Bradley–Terry** (pairwise): each item has a latent score `s_i`; 
  - `P(i ≻ j) = exp(s_i) / (exp(s_i) + exp(s_j))`.
- **Plackett–Luce** (listwise): probability distribution over full rankings.
- **Logistic regression on metric differences**: 
  - `P(A ≻ B) = σ(βᵀ (φ(A) − φ(B)))`,
  - where `φ(.)` are metric features; this directly learns a linear utility.

**Why it’s more realistic / stronger**
- If you have any real or simulated “click/choice” data, you can fit personas instead of hand-designing them.
- Produces probabilistic preferences and uncertainty estimates.

**Practical notes for HC-XAI**
- Start with logistic regression on metric deltas; it is easy, interpretable, and maps to an additive utility.
- Add regularization (L2/L1) to avoid unstable coefficients.
- Evaluate with held-out pairwise accuracy / log loss, and also induced ranking metrics (Kendall’s τ, NDCG).

**References (pairwise / ranking models)**
- Bradley, R. A., & Terry, M. E. (1952). *Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons.* Biometrika. DOI: `10.2307/2334029` https://doi.org/10.2307/2334029
- Plackett, R. L. (1975). *The Analysis of Permutations.* Applied Statistics. DOI: `10.2307/2346567` https://doi.org/10.2307/2346567
- Burges, C., et al. (2005). *Learning to rank using gradient descent.* ICML. DOI: `10.1145/1102351.1102363` https://doi.org/10.1145/1102351.1102363
- Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). *BPR: Bayesian Personalized Ranking from Implicit Feedback.* UAI. DBLP: https://dblp.org/rec/conf/uai/RendleFGS09 (also appears as arXiv/CoRR `abs/1205.2618` via DBLP: https://dblp.org/rec/journals/corr/abs-1205-2618 )
- Luce, R. D. *Individual Choice Behavior: A Theoretical Analysis.* (book; OpenLibrary work `https://openlibrary.org/works/OL4319279W`; example ISBN-13: `9780486153391`).
- McFadden, D. *Conditional logit analysis of qualitative choice behavior.* (canonical discrete-choice reference; OpenAlex work `https://openalex.org/W1550812296` shows author “Daniel McFadden” and very high citation count).

## 3) Mixture / Noisy User (Stochastic Personas)

**Idea**
- Users are inconsistent: the same person may choose differently due to attention limits, ambiguity, or context.
- Model persona as a mixture of:
  - a “core” deterministic preference (priority rule or weighted utility)
  - plus noise (random choices, Gumbel noise, logistic noise)

**Simple forms**
- **ε-noise mixture**: with probability `1−ε` follow the base preference; with probability `ε`, flip or pick randomly.
- **Random utility**: `U = deterministic_part + noise`, and choices follow a softmax/logit form.

**Why it helps**
- Avoids overconfident conclusions from perfectly consistent synthetic labels.
- Lets you test whether recommenders remain good when preferences are noisy.

**Practical notes for HC-XAI**
- If you use pairwise labels, you can inject noise by flipping a fraction of labels.
- Or generate labels from a probabilistic model: sample winner according to `P(A ≻ B)`.
- Report performance vs noise level `ε`.

**References (stochastic choice / noise models)**
- Thurstone, L. L. (1927). *A law of comparative judgment.* Psychological Review. DOI: `10.1037/h0070288` https://doi.org/10.1037/h0070288
- Luce, R. D. *Individual Choice Behavior: A Theoretical Analysis.* (book; OpenLibrary work `https://openlibrary.org/works/OL4319279W`).
- McFadden’s logit / random-utility family is commonly cited as the standard framework in discrete choice; see OpenAlex entry for “Conditional logit…”: `https://openalex.org/W1550812296`.

## 4) Context-Dependent Preferences (Personalization Beyond Global Weights)

**Idea**
- A persona’s tradeoffs can depend on the instance: e.g., in high-risk cases a regulator might prioritize faithfulness more.
- Model weights as a function of context `x` (instance features, model confidence, subgroup, etc.):
  - `w(x) = g(x; θ)` and `U(A|x) = w(x)ᵀ φ(A,x)`.

**Typical approaches**
- **Conditional logit / generalized linear models** with context features.
- **Hierarchical / mixed logit**: captures per-user random effects and heterogeneous tastes.
- **Contextual bandits** (online): learns preferences from interaction feedback.

**Why it’s closest to “human-centered” goals**
- Captures that users change what they care about based on scenario.
- Supports genuinely personalized recommendation policies.

**Practical notes for HC-XAI**
- Start simple: interact context with metric deltas (e.g., `βᵀ[(φ(A)−φ(B)) ⊗ context]`).
- Or cluster instances into “contexts” and fit separate weight vectors.
- Beware leakage: context features must be available at decision time.

**References (context dependence / personalization)** **(Need to verify)**
- Train, K. *Discrete Choice Methods with Simulation.* (Cambridge Univ. Press; Crossref-indexed chapter DOIs under `10.1017/CBO9780511753930.*`, e.g. `10.1017/CBO9780511753930.003` https://doi.org/10.1017/CBO9780511753930.003).
- Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). *A contextual-bandit approach to personalized news article recommendation.* WWW. DOI: `10.1145/1772690.1772758` https://doi.org/10.1145/1772690.1772758
- McFadden-style conditional logit is the standard “context-dependent utility” baseline in discrete choice; OpenAlex work for “Conditional logit…”: `https://openalex.org/W1550812296`.

---

# Implementation TODOs (Check Lists)

## A) Add persona presets beyond strict priorities
- [ ] Define persona specs as config entries (e.g., YAML): model type (priority / weighted / learned / noisy / context), metric groups, directions, normalization.
- [ ] Ensure every metric has an explicit direction and optional transform.

## B) Weighted-utility personas
- [ ] Implement a `WeightedUtilityPersona` that outputs per-instance rankings over method_variants.
- [ ] Add weight sweeps (grid + random Dirichlet) for sensitivity analysis.

## C) Learned personas from pairwise labels
- [ ] Implement logistic regression on metric deltas (baseline learned persona).
- [ ] Implement Bradley–Terry / Plackett–Luce for comparison.
- [ ] Add train/validation/test split by instance (avoid leakage across pairs).

## D) Noisy personas
- [ ] Add label-noise injection utility (flip fraction ε) and evaluate robustness.
- [ ] Add stochastic choice sampling from softmax over utilities.

## E) Context-dependent personas
- [ ] Define context features available at recommendation time (model confidence, predicted class, subgroup).
- [ ] Implement a context-aware preference model (conditional logit or interaction features).
- [ ] Compare against global-weight personas.
