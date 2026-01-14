python3 -m src.preference_learning.run_all \
  --experiment-mode autoxai-comparison \
  --autoxai-metric-mode auto-xai \
  --encoded-dir results/hc_combo_20260110_024805/encoded_pareto_fronts_new/features_full_lm_stats \
  --output-dir results/hc_combo_20260110_024805/preference_learning_new_c_200_tau_1e-2 \
  --exclude-feature-groups statistical landmarking \
  --num-users 40 \
  --tau 0.001 \
  --concentration-c 200 \
