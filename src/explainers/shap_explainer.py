"""
SHAP explainer for local (per-instance) explanations on TABULAR data only.
Depends on BaseExplainer from src.explainers.base.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional

from .base import BaseExplainer, InstanceLike, ArrayLike


class SHAPExplainer(BaseExplainer):
    """Minimal SHAP explainer for tabular data."""

    supported_data_types = ["tabular"]
    supported_model_types = ["sklearn", "xgboost", "lightgbm", "catboost", "generic-predict"]

    def __init__(self, config: Dict[str, Any], model: Any, dataset: Any):
        super().__init__(config, model, dataset)
        self._shap = None
        self._explainer = None
        self._is_tree = False
        self._background: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> None:
        """
        Prepare SHAP explainer.
        - Tree models: TreeExplainer(model)
        - Others: KernelExplainer(predict_fn, background)
        If SHAP is not installed, we'll fall back to a simple permutation scheme in explain_instance.
        """
        try:
            import shap  # type: ignore
            self._shap = shap
        except Exception:
            self._shap = None
            self.logger.warning("`shap` not available; will use permutation fallback.")
            return

        X_np, _ = self._coerce_X_y(X, None)
        self._is_tree = self._is_tree_model()

        if self._is_tree:
            # Exact fast SHAP for tree ensembles
            self._explainer = self._shap.TreeExplainer(self._underlying_model())
        else:
            # KernelSHAP with small background
            bsize = int(self._expl_cfg.get("background_sample_size", 100))
            bsize = min(bsize, len(X_np))
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X_np), size=bsize, replace=False)
            self._background = X_np[idx]
            predict_fn = self._kernel_predict_fn()
            self._explainer = self._shap.KernelExplainer(predict_fn, self._background)

    def explain_instance(self, instance: InstanceLike) -> Dict[str, Any]:
        """
        Produce a local explanation for one instance.
        Returns standardized dict as expected by BaseExplainer.
        """
        # If shap missing, use permutation fallback
        if self._shap is None or self._explainer is None:
            return self._explain_with_permutation(instance)

        inst2d = self._to_numpy_2d(instance)
        pred, t_pred = self._timeit(self._predict, inst2d)
        proba = self._predict_proba(inst2d)

        # Compute SHAP values
        if self._is_tree:
            shap_vals_raw, t_shap = self._timeit(self._explainer.shap_values, inst2d, silent=True)
            shap_vals = self._select_shap_values(shap_vals_raw, pred)
            expected = self._explainer.expected_value
            exp_val = self._select_expected_value(expected, pred)
        else:
            # KernelExplainer expects small batches; just one instance
            shap_vals_raw, t_shap = self._timeit(self._explainer.shap_values, inst2d,silent=True)
            shap_vals = self._select_shap_values(shap_vals_raw, pred)
            expected = self._explainer.expected_value
            exp_val = self._select_expected_value(expected, pred)

        # Standardize output
        feature_names = self._infer_feature_names(inst2d[0])
        result = self._standardize_explanation_output(
            attributions=np.asarray(shap_vals[0]).tolist() if shap_vals.ndim == 2 else np.asarray(shap_vals).tolist(),
            instance=inst2d[0],
            prediction=pred[0] if len(pred) else pred,
            prediction_proba=proba[0] if proba is not None and len(proba) else None,
            feature_names=feature_names,
            metadata={"expected_value": exp_val},
            per_instance_time=t_pred + t_shap,
        )
        return result

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _underlying_model(self):
        """Return raw model object (unwrap simple wrappers if present)."""
        return getattr(self.model, "model", self.model)

    def _is_tree_model(self) -> bool:
        """Heuristic: check common tree model class names."""
        name = type(self._underlying_model()).__name__.lower()
        return any(k in name for k in ["decisiontree", "randomforest", "gradientboost", "xgb", "lgbm", "lightgbm"])

    def _kernel_predict_fn(self):
        """Prediction function for KernelSHAP (classification-friendly)."""
        if hasattr(self.model, "predict_proba"):
            return lambda x: self.model.predict_proba(x)  # returns (n, C)
        return lambda x: self.model.predict(x)  # shape (n,) or (n, 1)

    def _select_shap_values(self, shap_values_raw, prediction: np.ndarray) -> np.ndarray:
        """
        Normalize SHAP outputs to shape (n_samples, n_features).
        Handles SHAP returning:
          - list of arrays per class,
          - 3D arrays (n, n_features, n_classes),
          - 2D arrays (n, n_features).
        For classification, uses the predicted class; for binary with list format, uses class 1.
        """
        # list-of-arrays => choose class
        if isinstance(shap_values_raw, list):
            if len(shap_values_raw) == 2:
                return np.asarray(shap_values_raw[1])  # positive class
            # multiclass: pick predicted class per row
            preds = prediction.astype(int).ravel()
            return np.vstack([np.asarray(shap_values_raw[c])[i] for i, c in enumerate(preds)])

        shap_values = np.asarray(shap_values_raw)

        # 3D (n, n_features, n_classes)
        if shap_values.ndim == 3:
            if shap_values.shape[2] == 2:
                return shap_values[:, :, 1]
            preds = prediction.astype(int).ravel()
            return np.vstack([shap_values[i, :, preds[i]] for i in range(len(preds))])

        # 2D already (n, n_features)
        return shap_values

    def _select_expected_value(self, expected_value, prediction: np.ndarray) -> float:
        """
        Expected value may be scalar or per-class array. Match selection used in _select_shap_values.
        """
        if isinstance(expected_value, (list, np.ndarray)):
            ev = np.asarray(expected_value)
            if ev.ndim == 0:
                return float(ev)
            if ev.size == 2:
                return float(ev[1])
            pred = int(np.asarray(prediction).ravel()[0])
            return float(ev[pred])
        return float(expected_value)

    # -------------------------------------------------------------------------
    # Permutation fallback (no SHAP installed)
    # -------------------------------------------------------------------------

    def _explain_with_permutation(self, instance: InstanceLike) -> Dict[str, Any]:
        """
        Very small, local permutation importance around the instance.
        Replaces each feature with a simple background statistic (mean) to measure impact.
        """
        inst = self._to_numpy_2d(instance)[0]
        # Background mean: try dataset statistics; fallback to zeros.
        bg_mean = getattr(self.dataset, "feature_means", None)
        if bg_mean is None:
            X_bg = getattr(self.dataset, "X_train", None)
            if X_bg is not None:
                bg_mean = np.mean(np.asarray(X_bg), axis=0)
            else:
                bg_mean = np.zeros_like(inst)

        base_pred = float(np.asarray(self._predict(inst)).ravel()[0])
        importances = np.zeros_like(inst, dtype=float)

        for j in range(len(inst)):
            perturbed = inst.copy()
            perturbed[j] = bg_mean[j]
            new_pred = float(np.asarray(self._predict(perturbed)).ravel()[0])
            importances[j] = abs(base_pred - new_pred)

        feature_names = self._infer_feature_names(inst)
        return self._standardize_explanation_output(
            attributions=importances.tolist(),
            instance=inst,
            prediction=base_pred,
            prediction_proba=None,
            feature_names=feature_names,
            metadata={"fallback": "permutation"},
            per_instance_time=0.0,
        )
