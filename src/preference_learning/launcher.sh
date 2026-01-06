#!/usr/bin/env bash
python3 -m src.preference_learning.run_all \
  --encoded-dir results/full_run_dec8/encoded_pareto_fronts/features_full_lm_stats \
  --output-dir results/full_run_dec8/preference_learning_simulation_advanced/basic_features/tuned-svc-basic_tau_and_users \
  --personas layperson regulator clinician \
  --exclude-feature-groups statistical landmarking \
  --tune-svc \
  --num-users 10 \
  --tau 0.05
