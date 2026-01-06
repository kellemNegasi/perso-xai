"""Persona models for sampling user preferences over explanation metrics.

This module implements hierarchical Dirichlet personas defined by JSON configs
in `src/preference_learning/configs/`. A persona samples:
1) group weights over metric groups, and
2) within-group weights over metrics,
then combines them into final metric weights.

These weights can be used to compute utilities over z-normalised metric vectors
and to sample pairwise preferences via a logistic (sigmoid) model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import yaml

# Metrics where "lower is better" in raw form; we flip sign to align with utility maximisation.
DEFAULT_NEGATE_METRICS: frozenset[str] = frozenset(
    {
        "infidelity",
        "non_sensitivity_violation_fraction",
        "non_sensitivity_delta_mean",
        "relative_input_stability",
        "covariate_complexity",
    }
)

DEFAULT_TAU: float = 0.05
DEFAULT_PREFERENCE_MODEL_PATH: Path = (
    Path(__file__).resolve().parent / "configs" / "preference_model.yml"
)
DEFAULT_METRIC_LEVEL_LAMBDA: float = 0.5
MIN_DIRICHLET_CONCENTRATION: float = 1e-6

_PREFERENCE_MODEL_CACHE: dict[Path, Mapping[str, object] | None] = {}


@dataclass(frozen=True)
class MetricGroupConfig:
    name: str
    alpha: float | None
    preference: float | None
    metrics: Mapping[str, float]

    def metric_names(self) -> Sequence[str]:
        return tuple(self.metrics.keys())


@dataclass(frozen=True)
class PersonaConfig:
    persona: str
    type: str
    description: str | None
    tau: float | None
    groups: Sequence[MetricGroupConfig]

    def metric_names(self) -> Sequence[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for group in self.groups:
            for metric in group.metrics.keys():
                if metric not in seen:
                    ordered.append(metric)
                    seen.add(metric)
        return tuple(ordered)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PersonaConfig":
        persona = payload.get("persona")
        persona_type = payload.get("type")
        if not isinstance(persona, str) or not persona:
            raise ValueError("Persona config missing non-empty 'persona' field.")
        if not isinstance(persona_type, str) or not persona_type:
            raise ValueError("Persona config missing non-empty 'type' field.")
        description = payload.get("description")
        if description is not None and not isinstance(description, str):
            description = None
        tau = payload.get("tau")
        if tau is None:
            parsed_tau: float | None = None
        elif isinstance(tau, (int, float)):
            if float(tau) <= 0:
                raise ValueError(f"Persona config has invalid tau={tau!r}; must be > 0.")
            parsed_tau = float(tau)
        else:
            raise ValueError(f"Persona config has invalid tau type: {type(tau)}")
        raw_groups = payload.get("groups")
        if not isinstance(raw_groups, Mapping) or not raw_groups:
            raise ValueError("Persona config missing non-empty 'groups' mapping.")

        groups: list[MetricGroupConfig] = []
        for group_name, group_payload in raw_groups.items():
            if not isinstance(group_name, str) or not group_name:
                continue
            if not isinstance(group_payload, Mapping):
                continue
            alpha = group_payload.get("alpha")
            preference = group_payload.get("preference")
            metrics = group_payload.get("metrics")

            parsed_alpha: float | None
            if alpha is None:
                parsed_alpha = None
            elif isinstance(alpha, (int, float)) and float(alpha) > 0:
                parsed_alpha = float(alpha)
            else:
                raise ValueError(f"Group '{group_name}' has invalid alpha={alpha!r}.")

            parsed_preference: float | None
            if preference is None:
                parsed_preference = None
            elif isinstance(preference, (int, float)) and float(preference) > 0:
                parsed_preference = float(preference)
            else:
                raise ValueError(f"Group '{group_name}' has invalid preference={preference!r}.")

            if parsed_alpha is None and parsed_preference is None:
                raise ValueError(
                    f"Group '{group_name}' must define either a positive 'alpha' (legacy) or 'preference' rating."
                )

            if not isinstance(metrics, Mapping) or not metrics:
                raise ValueError(f"Group '{group_name}' has no metrics mapping.")
            metric_alphas: Dict[str, float] = {}
            for metric_name, metric_alpha in metrics.items():
                if not isinstance(metric_name, str) or not metric_name:
                    continue
                if not isinstance(metric_alpha, (int, float)) or metric_alpha <= 0:
                    raise ValueError(
                        f"Metric '{metric_name}' in group '{group_name}' has invalid alpha={metric_alpha!r}."
                    )
                metric_alphas[metric_name] = float(metric_alpha)
            if not metric_alphas:
                raise ValueError(f"Group '{group_name}' contains no valid metrics.")
            groups.append(
                MetricGroupConfig(
                    name=group_name,
                    alpha=parsed_alpha,
                    preference=parsed_preference,
                    metrics=metric_alphas,
                )
            )

        if not groups:
            raise ValueError("Persona config contains no valid groups.")
        return cls(
            persona=persona,
            type=persona_type,
            description=description,
            tau=parsed_tau,
            groups=tuple(groups),
        )


def load_persona_config(path: Path) -> PersonaConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Persona config must be a JSON object. Got: {type(payload)}")
    return PersonaConfig.from_dict(payload)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Stable sigmoid implementation (avoids overflow for large |x|).
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def z_normalize_matrix(values: np.ndarray) -> np.ndarray:
    """Column-wise z-normalisation ignoring NaNs; NaNs become 0 after scaling."""
    if values.ndim != 2:
        raise ValueError("values must be a 2D array (n_candidates, n_metrics).")
    means = np.nanmean(values, axis=0)
    stds = np.nanstd(values, axis=0)
    stds = np.where(stds > 0, stds, 1.0)
    z = (values - means) / stds
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_positive(scores: Sequence[float], *, label: str) -> np.ndarray:
    arr = np.asarray(list(scores), dtype=float)
    if arr.size == 0:
        raise ValueError(f"{label} scores must be non-empty.")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError(f"{label} scores must be finite and > 0. Got: {arr!r}")
    total = float(arr.sum())
    if total <= 0:
        raise ValueError(f"{label} scores sum must be > 0.")
    return arr / total


def _load_preference_model_payload(path: Path = DEFAULT_PREFERENCE_MODEL_PATH) -> Mapping[str, object] | None:
    cached = _PREFERENCE_MODEL_CACHE.get(path)
    if cached is not None or path in _PREFERENCE_MODEL_CACHE:
        return cached
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _PREFERENCE_MODEL_CACHE[path] = None
        return None
    if not isinstance(raw, Mapping):
        _PREFERENCE_MODEL_CACHE[path] = None
        return None
    payload = raw.get("preference_model")
    if not isinstance(payload, Mapping):
        _PREFERENCE_MODEL_CACHE[path] = None
        return None
    _PREFERENCE_MODEL_CACHE[path] = payload
    return payload


def _lookup_metric_level_lambda(payload: Mapping[str, object] | None) -> float:
    if payload is None:
        return DEFAULT_METRIC_LEVEL_LAMBDA
    concentration = payload.get("concentration")
    if not isinstance(concentration, Mapping):
        return DEFAULT_METRIC_LEVEL_LAMBDA
    metric_level = concentration.get("metric_level")
    if not isinstance(metric_level, Mapping):
        return DEFAULT_METRIC_LEVEL_LAMBDA
    lam = metric_level.get("lambda")
    if isinstance(lam, (int, float)) and float(lam) > 0:
        return float(lam)
    return DEFAULT_METRIC_LEVEL_LAMBDA


def _lookup_concentration_c(payload: Mapping[str, object] | None, persona: str) -> float:
    if payload is None:
        raise ValueError(f"Missing preference model config at {DEFAULT_PREFERENCE_MODEL_PATH}.")
    concentration = payload.get("concentration")
    if not isinstance(concentration, Mapping):
        raise ValueError("preference_model.concentration must be a mapping.")
    fixed = concentration.get("fixed")
    if not isinstance(fixed, Mapping):
        raise ValueError("preference_model.concentration.fixed must be a mapping.")
    value = fixed.get(persona)
    if not isinstance(value, (int, float)) or float(value) <= 0:
        raise ValueError(
            f"preference_model.concentration.fixed is missing a positive value for persona={persona!r}."
        )
    return float(max(float(value), MIN_DIRICHLET_CONCENTRATION))


class HierarchicalDirichletUser:
    """
    Hierarchical persona that samples a two-level Dirichlet over metrics and then stays fixed.

    Weight sampling
    ---------------
    - Metrics are partitioned into groups; group weights g are drawn from Dirichlet(alpha_group).
    - Within each group, metric weights m are drawn from Dirichlet(alpha_metric).
    - The final weight for a metric is proportional to g[group] * m[metric]; if a metric appears
      in multiple groups, contributions are summed and then all metric weights are normalised.

    Utility and preference
    ----------------------
    - Utility for an explanation e in context x: U(e|x) = sum_j w_j * z_j(metric_j(e, x)),
      where z_j is the z-normalised (and sign-flipped if needed) value of metric j.
    - Preference between candidates i and j with feature vectors phi_i, phi_j:
        P(e_i ≻ e_j) = sigmoid( (w^T (phi_i - phi_j)) / tau ),
      where phi_k = [z_1(metric_1(e_k)), ..., z_d(metric_d(e_k))] and tau is a temperature.

    Parameters
    ----------
    config : PersonaConfig
        Parsed persona definition containing either group/metric alphas (legacy) or 1–5 preference ratings.
    seed : int | None, optional
        RNG seed for reproducible weight sampling; if None, uses a nondeterministic seed.
    tau : float | None, optional
        Preference temperature; overrides config.tau when provided (must be > 0).
    negate_metrics : Iterable[str], optional
        Metrics where lower is better; their values are negated before normalisation.
    """

    def __init__(
        self,
        config: PersonaConfig,
        *,
        seed: int | None = None,
        tau: float | None = None,
        negate_metrics: Iterable[str] = DEFAULT_NEGATE_METRICS,
        metrics_already_oriented: bool = False,
    ) -> None:
        resolved_tau = float(DEFAULT_TAU if tau is None else tau)
        if config.tau is not None and tau is None:
            resolved_tau = float(config.tau)
        if resolved_tau <= 0:
            raise ValueError("tau must be > 0.")
        self.config = config
        self.tau = float(resolved_tau)
        self.negate_metrics = frozenset() if metrics_already_oriented else frozenset(negate_metrics)
        self._rng = np.random.default_rng(seed)
        self.metric_order: tuple[str, ...] = tuple(config.metric_names())
        self.metric_weights: Dict[str, float] = {}
        self.group_weights: Dict[str, float] = {}
        self._weight_vector: np.ndarray | None = None
        self.resample_weights()

    @property
    def weight_vector(self) -> np.ndarray:
        if self._weight_vector is None:
            self._weight_vector = np.asarray(
                [self.metric_weights.get(name, 0.0) for name in self.metric_order],
                dtype=float,
            )
        return self._weight_vector

    def resample_weights(self) -> None:
        """(Re)sample hierarchical Dirichlet weights and cache as a metric vector."""
        groups = list(self.config.groups)
        uses_preferences = any(group.preference is not None for group in groups)
        uses_alphas = any(group.alpha is not None for group in groups)
        if uses_preferences and uses_alphas:
            raise ValueError("Persona config mixes 'alpha' and 'preference' groups; choose one scheme.")

        if uses_preferences:
            group_scores = [float(group.preference) for group in groups if group.preference is not None]
            if len(group_scores) != len(groups):
                raise ValueError("Persona config is missing group 'preference' ratings for some groups.")
            w0_group = _normalize_positive(group_scores, label="Group preference")
            model_payload = _load_preference_model_payload()
            c = _lookup_concentration_c(model_payload, self.config.persona)
            group_alphas = np.maximum(c * w0_group, MIN_DIRICHLET_CONCENTRATION)
            group_weights = self._rng.dirichlet(group_alphas)
            self.group_weights = {group.name: float(w) for group, w in zip(groups, group_weights)}
            metric_lambda = _lookup_metric_level_lambda(model_payload)
            metric_concentration = float(max(metric_lambda * c, MIN_DIRICHLET_CONCENTRATION))
        else:
            group_alphas = np.asarray(
                [float(group.alpha) for group in groups if group.alpha is not None],
                dtype=float,
            )
            group_weights = self._rng.dirichlet(group_alphas)
            self.group_weights = {group.name: float(w) for group, w in zip(groups, group_weights)}
            metric_concentration = 0.0

        metric_weights: MutableMapping[str, float] = {}
        for group, group_w in zip(groups, group_weights):
            metric_names = list(group.metrics.keys())
            if uses_preferences:
                v0 = _normalize_positive(
                    [float(group.metrics[name]) for name in metric_names],
                    label=f"Metric preference for group '{group.name}'",
                )
                metric_alphas = np.maximum(metric_concentration * v0, MIN_DIRICHLET_CONCENTRATION)
            else:
                metric_alphas = np.asarray([group.metrics[name] for name in metric_names], dtype=float)
            within = self._rng.dirichlet(metric_alphas)
            for metric_name, metric_w in zip(metric_names, within):
                metric_weights[metric_name] = metric_weights.get(metric_name, 0.0) + float(
                    group_w * metric_w
                )

        total = float(sum(metric_weights.values()))
        if total <= 0:
            raise RuntimeError("Sampled metric weights sum to 0; check persona config alphas.")
        self.metric_weights = {k: v / total for k, v in metric_weights.items()}
        self._weight_vector = None

    def transform_metric_value(self, metric_name: str, value: float) -> float:
        return -float(value) if metric_name in self.negate_metrics else float(value)

    def vectorize_metrics(self, metrics: Mapping[str, object]) -> np.ndarray:
        """Convert a raw metric mapping into a dense vector aligned with `metric_order`."""
        vec = np.full((len(self.metric_order),), np.nan, dtype=float)
        for idx, name in enumerate(self.metric_order):
            raw = metrics.get(name)
            if isinstance(raw, (int, float)):
                vec[idx] = self.transform_metric_value(name, float(raw))
        return vec

    def utilities(self, z_matrix: np.ndarray) -> np.ndarray:
        """Compute U for each candidate: U = sum_j w_j * z_j."""
        if z_matrix.ndim != 2 or z_matrix.shape[1] != len(self.metric_order):
            raise ValueError("z_matrix must be (n_candidates, n_metrics) matching metric_order.")
        return z_matrix @ self.weight_vector

    def preference_probability(self, z_i: np.ndarray, z_j: np.ndarray) -> float:
        """Compute P(e_i ≻ e_j) = sigmoid(w^T(z_i - z_j) / tau)."""
        if z_i.shape != z_j.shape or z_i.shape != (len(self.metric_order),):
            raise ValueError("z_i and z_j must be 1D vectors aligned with metric_order.")
        delta = float(np.dot(self.weight_vector, (z_i - z_j)))
        return float(_sigmoid(np.asarray([delta / self.tau]))[0])

    def sample_preference(self, z_i: np.ndarray, z_j: np.ndarray) -> bool:
        """Bernoulli sample: returns 1 with prob sigmoid(w·(z_i - z_j)/tau), else 0."""
        p = self.preference_probability(z_i, z_j) # compute probability
        return bool(self._rng.random() < p)
