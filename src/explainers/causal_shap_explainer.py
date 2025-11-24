"""
Causal SHAP explainer for tabular data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import ArrayLike, BaseExplainer, InstanceLike


class CausalSHAPExplainer(BaseExplainer):
    """
    Approximate SHAP values while respecting a simple causal ordering inferred
    from feature correlations. Stub intended for tabular data only.
    """

    supported_data_types = ["tabular"]
    supported_model_types = [
        "sklearn",
        "xgboost",
        "lightgbm",
        "catboost",
        "generic-predict",
    ]

    def __init__(self, config: Dict[str, Any], model: Any, dataset: Any):
        super().__init__(config=config, model=model, dataset=dataset)
        self._X_train: Optional[np.ndarray] = getattr(dataset, "X_train", None)
        self._y_train: Optional[np.ndarray] = getattr(dataset, "y_train", None)

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> None:
        self._X_train, self._y_train = self._coerce_X_y(X, y)

    def explain_instance(self, instance: InstanceLike) -> Dict[str, Any]:
        inst2d = self._to_numpy_2d(instance)
        inst_vec = inst2d[0]

        X_train = self._ensure_training_data(inst_vec)
        feature_names = self._infer_feature_names(inst_vec)

        causal_graph = self._infer_causal_structure(X_train, feature_names)
        attributions, info = self._causal_shap(inst_vec, X_train, causal_graph, feature_names)

        prediction, t_pred = self._timeit(self._predict, inst2d)
        prediction_proba = self._predict_proba(inst2d)

        pred_arr = np.asarray(prediction).ravel()
        pred_value = float(pred_arr[0]) if pred_arr.size else float(pred_arr)

        proba_value = None
        if prediction_proba is not None:
            proba_value = np.asarray(prediction_proba)[0]

        metadata = {
            "causal_graph": causal_graph,
            "coalition_samples": info["coalition_samples"],
            "correlation_threshold": info["correlation_threshold"],
        }

        return self._standardize_explanation_output(
            attributions=attributions.tolist(),
            instance=inst_vec,
            prediction=pred_value,
            prediction_proba=proba_value,
            feature_names=feature_names,
            metadata=metadata,
            per_instance_time=t_pred,
        )
    
    def explain_batch(self, X: ArrayLike) -> List[Dict[str, Any]]:
        """
        Batch wrapper that reuses shared predictions while each instance still
        runs its causal-SHAP sampling.
        """
        X_np, _ = self._coerce_X_y(X, None)
        preds = np.asarray(self._predict(X_np))
        proba = self._predict_proba(X_np)

        results: List[Dict[str, Any]] = []
        for idx, inst_vec in enumerate(X_np):
            X_train = self._ensure_training_data(inst_vec)
            feature_names = self._infer_feature_names(inst_vec)
            attributions, info = self._causal_shap(inst_vec, X_train, feature_names, feature_names)

            pred_row = np.asarray(preds[idx]).ravel()
            pred_value = float(pred_row[0]) if pred_row.size else float(pred_row)

            proba_value = None
            if proba is not None:
                proba_value = np.asarray(proba[idx])

            metadata = {
                "causal_graph": info["causal_graph"] if "causal_graph" in info else self._infer_causal_structure(X_train, feature_names),
                "coalition_samples": info["coalition_samples"],
                "correlation_threshold": info["correlation_threshold"],
            }

            results.append(
                self._standardize_explanation_output(
                    attributions=attributions.tolist(),
                    instance=inst_vec,
                    prediction=pred_value,
                    prediction_proba=proba_value,
                    feature_names=feature_names,
                    metadata=metadata,
                    per_instance_time=0.0,
                )
            )
        return results

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_training_data(self, fallback_instance: np.ndarray) -> np.ndarray:
        if self._X_train is not None:
            return self._X_train
        self._X_train = fallback_instance.reshape(1, -1)
        return self._X_train

    def _infer_causal_structure(
        self, X_train: np.ndarray, feature_names: List[str]
    ) -> Dict[str, List[str]]:
        corr_threshold = float(self._expl_cfg.get("causal_shap_corr_threshold", 0.3))
        corr = np.corrcoef(X_train.T)
        graph: Dict[str, List[str]] = {}

        for i, fname in enumerate(feature_names):
            parents: List[str] = []
            for j in range(len(feature_names)):
                if i == j:
                    continue
                if abs(corr[i, j]) >= corr_threshold and j < i:
                    parents.append(feature_names[j])
            graph[fname] = parents
        return graph

    def _causal_shap(
        self,
        instance: np.ndarray,
        X_train: np.ndarray,
        causal_graph: Dict[str, List[str]],
        feature_names: List[str],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        baseline = np.mean(X_train, axis=0)
        base_pred = float(np.asarray(self._predict(instance.reshape(1, -1))).ravel()[0])
        baseline_pred = float(np.asarray(self._predict(baseline.reshape(1, -1))).ravel()[0])

        n_features = len(instance)
        coalition_samples = int(self._expl_cfg.get("causal_shap_coalitions", 50))

        contributions = np.zeros(n_features)
        for idx, fname in enumerate(feature_names):
            parents = causal_graph.get(fname, [])
            parent_indices = [feature_names.index(p) for p in parents if p in feature_names]
            marginal_effects = []

            for _ in range(coalition_samples):
                coalition = self._sample_coalition(idx, n_features, parent_indices)
                inst_without = baseline.copy()
                inst_without[coalition] = instance[coalition]
                pred_without = float(np.asarray(self._predict(inst_without.reshape(1, -1))).ravel()[0])

                inst_with = inst_without.copy()
                inst_with[idx] = instance[idx]
                pred_with = float(np.asarray(self._predict(inst_with.reshape(1, -1))).ravel()[0])

                marginal_effects.append(pred_with - pred_without)

            contributions[idx] = np.mean(marginal_effects)

        current_sum = contributions.sum()
        total_effect = base_pred - baseline_pred
        if abs(current_sum) > 1e-12:
            contributions *= total_effect / current_sum

        info = {
            "coalition_samples": coalition_samples,
            "correlation_threshold": float(self._expl_cfg.get("causal_shap_corr_threshold", 0.3)),
        }
        return contributions, info

    def _sample_coalition(
        self,
        feature_idx: int,
        n_features: int,
        parent_indices: List[int],
    ) -> List[int]:
        rng = np.random.default_rng(self.random_state)
        coalition_size = rng.integers(low=0, high=max(1, n_features - 1))
        coalition = []
        for parent in parent_indices:
            if rng.random() < 0.8:
                coalition.append(parent)

        others = [i for i in range(n_features) if i != feature_idx and i not in coalition]
        rng.shuffle(others)
        coalition.extend(others[: max(0, coalition_size - len(coalition))])
        return coalition
