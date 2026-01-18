"""Persona models for sampling user preferences over explanation metrics.

This module implements flat Dirichlet personas defined by YAML configs in
`src/preference_learning/configs/`. A persona samples weights directly over
metrics from property-level ratings (one rating shared by all metrics in a
property).

These weights can be used to compute utilities over z-normalised metric vectors
and to sample pairwise preferences via a logistic (sigmoid) model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

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
MIN_DIRICHLET_CONCENTRATION: float = 1e-6

_PREFERENCE_MODEL_CACHE: dict[Path, Mapping[str, object] | None] = {}


@dataclass(frozen=True)
class PropertyConfig:
    name: str
    preference: float
    metrics: Sequence[str]

    def metric_names(self) -> Sequence[str]:
        return tuple(self.metrics)


@dataclass(frozen=True)
class PersonaConfig:
    persona: str
    type: str
    description: str | None
    tau: float | None
    properties: Sequence[PropertyConfig]

    def metric_names(self) -> Sequence[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for prop in self.properties:
            for metric in prop.metrics:
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
        raw_properties = payload.get("properties")
        if not isinstance(raw_properties, Mapping) or not raw_properties:
            raise ValueError("Persona config missing non-empty 'properties' mapping.")

        properties: list[PropertyConfig] = []
        seen_metrics: set[str] = set()
        for prop_name, prop_payload in raw_properties.items():
            if not isinstance(prop_name, str) or not prop_name:
                continue
            if not isinstance(prop_payload, Mapping):
                continue
            preference = prop_payload.get("preference")
            metrics = prop_payload.get("metrics")

            if not isinstance(preference, (int, float)) or float(preference) <= 0:
                raise ValueError(f"Property '{prop_name}' has invalid preference={preference!r}.")

            if not isinstance(metrics, Sequence) or isinstance(metrics, (str, bytes)) or not metrics:
                raise ValueError(f"Property '{prop_name}' has no metrics list.")
            metric_names: list[str] = []
            for metric_name in metrics:
                if not isinstance(metric_name, str) or not metric_name:
                    raise ValueError(f"Property '{prop_name}' has invalid metric name={metric_name!r}.")
                if metric_name in seen_metrics:
                    raise ValueError(
                        f"Metric '{metric_name}' is listed under multiple properties; metrics must be unique."
                    )
                metric_names.append(metric_name)
                seen_metrics.add(metric_name)
            if not metric_names:
                raise ValueError(f"Property '{prop_name}' contains no valid metrics.")
            properties.append(
                PropertyConfig(
                    name=prop_name,
                    preference=float(preference),
                    metrics=tuple(metric_names),
                )
            )

        if not properties:
            raise ValueError("Persona config contains no valid properties.")
        return cls(
            persona=persona,
            type=persona_type,
            description=description,
            tau=parsed_tau,
            properties=tuple(properties),
        )


def load_persona_config(path: Path) -> PersonaConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Persona config must be a YAML mapping. Got: {type(payload)}")
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


# TODO: Rename to `FlatDirichletUser` (and keep a backwards-compatible alias) once the
# public API can change. The current name is legacy from an earlier hierarchical variant.
class HierarchicalDirichletUser:
    """
    Note: despite the class name, this implementation is *flat* (a single Dirichlet over all
    metrics). The "Hierarchical" prefix is legacy from an earlier hierarchical implementation.

    Flat persona that samples a single Dirichlet over metrics and then stays fixed.

    Weight sampling
    ---------------
    - Each property carries a 1–5 preference rating.
    - The rating is repeated for each metric in the property, normalised across metrics, and
      scaled by concentration c to form Dirichlet alphas.
    - Metric weights are drawn directly from this single Dirichlet distribution.

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
        Parsed persona definition containing property-level preference ratings.
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
        concentration_c: float | None = None,
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
        self.concentration_c_override = float(concentration_c) if concentration_c is not None else None
        self.concentration_c: float | None = None
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
        """(Re)sample flat Dirichlet weights and cache as a metric vector."""
        properties = list(self.config.properties)
        metric_names: list[str] = []
        metric_scores: list[float] = []
        for prop in properties:
            for metric_name in prop.metrics:
                metric_names.append(metric_name)
                metric_scores.append(float(prop.preference))

        w0_metric = _normalize_positive(metric_scores, label="Metric preference")
        model_payload = _load_preference_model_payload()
        c = (
            float(self.concentration_c_override)
            if self.concentration_c_override is not None
            else _lookup_concentration_c(model_payload, self.config.persona)
        )
        if c <= 0:
            raise ValueError("concentration_c must be > 0.")
        self.concentration_c = float(max(c, MIN_DIRICHLET_CONCENTRATION))
        metric_alphas = np.maximum(self.concentration_c * w0_metric, MIN_DIRICHLET_CONCENTRATION)
        sampled = self._rng.dirichlet(metric_alphas)
        self.metric_weights = {name: float(weight) for name, weight in zip(metric_names, sampled)}
        self.group_weights = {
            prop.name: float(sum(self.metric_weights[name] for name in prop.metrics))
            for prop in properties
        }
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
