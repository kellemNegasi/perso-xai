from __future__ import annotations
from typing import Any, Dict
from .shap_explainer import SHAPExplainer
from .lime_explainer import LIMEExplainer
from .integrated_gradients_explainer import IntegratedGradientsExplainer
from .causal_shap_explainer import CausalSHAPExplainer
from .example_based_explainer import PrototypeExplainer, CounterfactualExplainer
_NAME2CLS = {
    "shap": SHAPExplainer,
    "lime": LIMEExplainer,
    "integrated_gradients": IntegratedGradientsExplainer,
    "causal_shap": CausalSHAPExplainer,
    "prototype": PrototypeExplainer,
    "counterfactual": CounterfactualExplainer,
}


def make_explainer(config: Dict[str, Any], model: Any, dataset: Any):
    typ = (config.get("type") or "shap").lower()
    if typ not in _NAME2CLS:
        raise ValueError(f"Unknown explainer type: {typ}")
    return _NAME2CLS[typ](config=config, model=model, dataset=dataset)
