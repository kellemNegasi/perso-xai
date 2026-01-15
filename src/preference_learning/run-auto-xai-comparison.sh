python3 -m src.preference_learning.run_all \
  --experiment-mode autoxai-comparison \
  --autoxai-metric-mode auto-xai \
  --encoded-dir results/hc_combo_20260114_132532/encoded_pareto_fronts_new/features_full_lm_stats \
  --output-dir results/hc_combo_20260114_132532/preference_learning_new \
  --exclude-feature-groups statistical landmarking \
  --num-users 40 \
  --tau 0.05 \