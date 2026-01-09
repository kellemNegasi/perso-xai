from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    from skopt import Optimizer  # type: ignore
    from skopt.space import Categorical, Integer  # type: ignore
except Exception:  # pragma: no cover - skopt only if installed
    Optimizer = None
    Categorical = None
    Integer = None


@dataclass
class RandIntSpec:
    low: int
    high: int  # inclusive


@dataclass
class SearchSpace:
    categorical: Dict[str, Sequence[object]]
    randint: Dict[str, RandIntSpec]


class BayesRangeOptimizer:
    """
    Minimal Bayesian optimizer over mixed categorical + randint (integer) spaces.

    Uses scikit-optimize's `Optimizer` under the hood. The caller is responsible for
    evaluating suggested params and calling `tell(params, score)`.
    """

    def __init__(
        self,
        *,
        space: SearchSpace,
        seed: int = 0,
        n_initial_points: int = 5,
    ) -> None:
        if Optimizer is None or Categorical is None or Integer is None:  # pragma: no cover
            raise RuntimeError(
                "Bayesian optimization requires scikit-optimize. Install it with `pip install scikit-optimize`."
            )
        self._space = space
        self._seed = int(seed)
        self._n_initial_points = max(1, int(n_initial_points))

        keys: List[str] = []
        dims: List[object] = []
        for key in sorted(space.categorical.keys()):
            keys.append(key)
            dims.append(Categorical(list(space.categorical[key]), name=key))
        for key in sorted(space.randint.keys()):
            keys.append(key)
            spec = space.randint[key]
            dims.append(Integer(int(spec.low), int(spec.high), name=key))

        self._keys = keys
        self._opt = Optimizer(
            dims,
            random_state=self._seed,
            n_initial_points=self._n_initial_points,
        )

    @property
    def keys(self) -> Sequence[str]:
        return tuple(self._keys)

    def ask(self) -> Dict[str, Any]:
        values = self._opt.ask()
        return {key: value for key, value in zip(self._keys, values)}

    def tell(self, params: Mapping[str, Any], score: float) -> None:
        # skopt minimizes; we maximize score.
        x = [params[key] for key in self._keys]
        self._opt.tell(x, -float(score))

