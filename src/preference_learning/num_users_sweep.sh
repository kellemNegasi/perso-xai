#!/usr/bin/env bash
python3 -m src.preference_learning.run_all \
  --encoded-dir results/full_run_dec8/encoded_pareto_fronts/features_full_lm_stats \
  --personas layperson regulator clinician \
  --exclude-feature-groups statistical landmarking \
  --tau 0.01 \
  --num-users-results-dir results/full_run_dec8/preference_learning_simulation/basic_features/untuned-svc/num_users_sweep \
  --num-users-values 3 5 10 20 30 40
