from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Sequence, TypeVar, Callable
import random

from .custom_types import DensityVector
from .generators import Generator
from .adjoint import AdjointGenerator

DensityLike = TypeVar("DensityLike")
LawLike = TypeVar("LawLike")
X = TypeVar("X")

@dataclass(slots=True)
class StationaryDistributionSolver(Generic[X]):
    """
    Solve A^{*} p_{\infty} = 0 by averaging forward density evolution.
    First check the Foster-Lyapunov drift condition to heuristically guess if the stationary distribution would even exist.
    """

    adjoint: AdjointGenerator[DensityVector, Any]
    states: Sequence[X]
    dt: float = 0.05

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        if not self.states:
            raise ValueError("states must be nonempty")

    def _step(self, p: DensityVector) -> list[float]:
        adjoint_p = self.adjoint.apply_to_density(p)
        updated = [pi + self.dt * dpi for pi, dpi in zip(p, adjoint_p)]
        return _normalize_density_vector(updated)
    
    def check_foster_lyapunov(
        self,
        generator: Generator[X],
        lyapunov_f: Callable[[X], float] | None = None,
        *,
        f_threshold: float | None = None,
    ) -> FosterLyapunovResult:
        """Heuristic Foster-Lyapunov drift check for f=||x||^2 (or custom f).

        Uses the generator A applied to f and checks for negative drift
        on a high-f subset of the state space.
        """
        lyapunov_f = lyapunov_f or _default_quadratic_lyapunov
        f_values = [float(lyapunov_f(state)) for state in self.states]
        if any(value < 0 for value in f_values):
            return FosterLyapunovResult(
                holds=False,
                c=None,
                b=None,
                threshold=None,
                message="Lyapunov function must be nonnegative.",
            )
        threshold = f_threshold if f_threshold is not None else _quantile(f_values, 0.75)
        region = [
            i
            for i, value in enumerate(f_values)
            if value >= threshold and value > 0.0
        ]
        if not region:
            return FosterLyapunovResult(
                holds=False,
                c=None,
                b=None,
                threshold=threshold,
                message="No states above threshold for drift check.",
            )
        generator_values = [
            float(generator.estimate_Af(lyapunov_f, state)) for state in self.states
        ]
        if any(generator_values[i] >= 0.0 for i in region):
            return FosterLyapunovResult(
                holds=False,
                c=None,
                b=None,
                threshold=threshold,
                message="Nonnegative drift detected on high-f region.",
            )
        c_candidates = [-generator_values[i] / f_values[i] for i in region]
        c = min(c_candidates)
        if c <= 0.0:
            return FosterLyapunovResult(
                holds=False,
                c=None,
                b=None,
                threshold=threshold,
                message="Unable to find positive drift rate c.",
            )
        b = max(generator_values[i] + c * f_values[i] for i in range(len(f_values)))
        b = max(0.0, float(b))
        return FosterLyapunovResult(
            holds=True,
            c=float(c),
            b=b,
            threshold=threshold,
            message="Foster-Lyapunov drift condition holds on high-f region.",
        )


    def solve_truncated(
        self,
        p0: DensityVector,
        *,
        lam: float,
        n_steps: int,
        check_foster_lyapunov: bool = False,
        generator: Generator[X] | None = None,
        lyapunov_f: Callable[[X], float] | None = None,
        f_threshold: float | None = None,
    ) -> list[float]:
        """Truncated geometric-weighted average of densities."""
        if not (0.0 < lam < 1.0):
            raise ValueError("lam must be in (0, 1)")
        if n_steps <= 0:
            raise ValueError("n_steps must be > 0")
        if len(p0) != len(self.states):
            raise ValueError("initial density length must match states")
        if check_foster_lyapunov:
            if generator is None:
                raise ValueError("generator must be provided for drift check")
            result = self.check_foster_lyapunov(
                generator, lyapunov_f=lyapunov_f, f_threshold=f_threshold
            )
            if not result.holds:
                raise ValueError(
                    "Foster-Lyapunov drift check failed: " + result.message
                )
        p = _normalize_density_vector(p0)
        total = [0.0 for _ in range(len(self.states))]
        weight = 1.0
        for _ in range(n_steps):
            for i in range(len(self.states)):
                total[i] += weight * p[i]
            p = self._step(p)
            weight *= lam
        return _normalize_density_vector([(1.0 - lam) * ti for ti in total])

    def solve_geometric(
        self,
        p0: DensityVector,
        *,
        lam: float,
        check_foster_lyapunov: bool = False,
        generator: Generator[X] | None = None,
        lyapunov_f: Callable[[X], float] | None = None,
        f_threshold: float | None = None,
        rng: random.Random | None = None,
        seed: int | None = None,
    ) -> list[float]:
        """Unbiased geometric-horizon estimator for weighted infinite sum."""
        if not (0.0 < lam < 1.0):
            raise ValueError("lam must be in (0, 1)")
        if len(p0) != len(self.states):
            raise ValueError("initial density length must match states")
        if check_foster_lyapunov:
            if generator is None:
                raise ValueError("generator must be provided for drift check")
            result = self.check_foster_lyapunov(
                generator, lyapunov_f=lyapunov_f, f_threshold=f_threshold
            )
            if not result.holds:
                raise ValueError(
                    "Foster-Lyapunov drift check failed: " + result.message
                )
        rng = rng or random.Random(seed)
        p = _normalize_density_vector(p0)
        total = [0.0 for _ in range(len(self.states))]
        while True:
            for i in range(len(self.states)):
                total[i] += p[i]
            if rng.random() > lam:
                break
            p = self._step(p)
        return _normalize_density_vector([(1.0 - lam) * ti for ti in total])
    
def _normalize_density_vector(p: DensityVector) -> list[float]:
    total = float(sum(p))
    if total <= 0:
        raise ValueError("density must have positive total mass")
    return [float(pi) / total for pi in p]

def _default_quadratic_lyapunov(x: X) -> float:
    if isinstance(x, (int, float)):
        return float(x) ** 2
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
        return sum(float(v) ** 2 for v in x)
    raise TypeError("Provide lyapunov_f for non-numeric states.")


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("values must be nonempty")
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(values)
    idx = int(q * (len(ordered) - 1))
    return ordered[idx]


@dataclass(slots=True)
class FosterLyapunovResult:
    holds: bool
    c: float | None
    b: float | None
    threshold: float | None
    message: str