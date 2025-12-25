from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml


def load_explainer_grid(config_path: Path) -> Dict[str, Dict[str, List[object]]]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    explainers = payload.get("explainers") if isinstance(payload, dict) else None
    if not isinstance(explainers, dict):
        raise ValueError(f"Unexpected grid format in {config_path}.")
    parsed: Dict[str, Dict[str, List[object]]] = {}
    for method, grid in explainers.items():
        if not isinstance(grid, dict):
            continue
        parsed[method] = {key: list(values) for key, values in grid.items() if isinstance(values, list)}
    return parsed


def load_default_variants(explainers_config_path: Path) -> Dict[str, str]:
    """
    Extract default method_variant identifiers from `src/configs/explainers.yml`.

    This matches the naming scheme used in HC-XAI metric artifacts, e.g.:
      - lime__lime_kernel_width-2.0__lime_num_samples-100
      - shap__background_sample_size-100
    """
    payload = yaml.safe_load(explainers_config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected explainers.yml format in {explainers_config_path}.")

    defaults: Dict[str, str] = {}
    lime = payload.get("lime")
    if isinstance(lime, dict):
        params = (
            ((lime.get("params") or {}).get("experiment") or {}).get("explanation")
            if isinstance(lime.get("params"), dict)
            else None
        )
        if isinstance(params, dict):
            kw = params.get("lime_kernel_width")
            ns = params.get("lime_num_samples")
            if kw is not None and ns is not None:
                defaults["lime"] = f"lime__lime_kernel_width-{kw}__lime_num_samples-{ns}"

    shap_cfg = payload.get("shap")
    if isinstance(shap_cfg, dict):
        params = (
            ((shap_cfg.get("params") or {}).get("experiment") or {}).get("explanation")
            if isinstance(shap_cfg.get("params"), dict)
            else None
        )
        if isinstance(params, dict):
            bg = params.get("background_sample_size")
            if bg is not None:
                defaults["shap"] = f"shap__background_sample_size-{bg}"

    ig_cfg = payload.get("integrated_gradients")
    if isinstance(ig_cfg, dict):
        params = (
            ((ig_cfg.get("params") or {}).get("experiment") or {}).get("explanation")
            if isinstance(ig_cfg.get("params"), dict)
            else None
        )
        if isinstance(params, dict):
            steps = params.get("ig_steps")
            if steps is not None:
                defaults["integrated_gradients"] = f"integrated_gradients__ig_steps-{steps}"

    causal_cfg = payload.get("causal_shap")
    if isinstance(causal_cfg, dict):
        params = (
            ((causal_cfg.get("params") or {}).get("experiment") or {}).get("explanation")
            if isinstance(causal_cfg.get("params"), dict)
            else None
        )
        if isinstance(params, dict):
            coalitions = params.get("causal_shap_coalitions")
            if coalitions is not None:
                defaults["causal_shap"] = f"causal_shap__causal_shap_coalitions-{coalitions}"

    return defaults


def method_metrics_path(results_root: Path, dataset: str, model: str, method: str) -> Path:
    return results_root / "metrics_results" / dataset / model / f"{method}_metrics.json"

