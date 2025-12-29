#!/usr/bin/env bash
python3 -m src.preference_learning.run_all \
  --encoded-dir results/full_run_dec8/encoded_pareto_fronts/features_full_lm_stats \
  --personas layperson regulator clinician \
  --exclude-feature-groups statistical landmarking \
  --tau-results-dir results/full_run_dec8/preference_learning_simulation/basic_features/untuned-svc/tau_tuning \
  --tau-values 0.01 0.03 0.05 0.1 0.2
