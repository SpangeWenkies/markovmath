from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Protocol, Sequence, TypeVar
import random

from core_interfaces import Density
from .custom_types import DensityVector

DensityLike = TypeVar("DensityLike")
LawLike = TypeVar("LawLike")
X = TypeVar("X")


class AdjointGenerator(Protocol[DensityLike, LawLike]):
    """Adjoint generator interface for forward evolution of laws/densities."""

    def apply_to_density(self, p: DensityLike) -> DensityLike: ...
    def apply_to_law(self, mu: LawLike) -> LawLike: ...


def _central_diff(f: Callable[[float], float], x: float, h: float) -> float:
    return (f(x + h) - f(x - h)) / (2.0 * h)


def _central_second_diff(f: Callable[[float], float], x: float, h: float) -> float:
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h**2)


@dataclass(slots=True)
class ClosedFormDiffusionAdjoint1D(AdjointGenerator[Density[float], Any]):
    """Adjoint generator for 1D diffusions using integration by parts.

    For generator A f = b f' + 0.5 a f'', the adjoint is:
        A* p = -d/dx(b p) + 0.5 d^2/dx^2(a p).
    """

    drift: Callable[[float], float]
    diffusion_coeff: Callable[[float], float]
    fd_step: float = 1e-4

    def __post_init__(self) -> None:
        if self.fd_step <= 0:
            raise ValueError("fd_step must be > 0")

    def apply_to_density(self, p: Density[float]) -> Density[float]:
        def adjoint_p(x: float) -> float:
            h = self.fd_step
            drift_term = _central_diff(lambda y: self.drift(y) * p(y), x, h)
            diffusion_term = _central_second_diff(
                lambda y: (self.diffusion_coeff(y) ** 2) * p(y), x, h
            )
            return -drift_term + 0.5 * diffusion_term

        return adjoint_p

    def apply_to_law(self, mu: Any) -> Any:
        raise NotImplementedError(
            "Adjoint diffusion requires a density representation."
        )


def ou_adjoint_1d(
    kappa: float, theta: float, sigma: float, *, fd_step: float = 1e-4
) -> ClosedFormDiffusionAdjoint1D:
    """OU example: dX = kappa(theta - X) dt + sigma dW."""
    return ClosedFormDiffusionAdjoint1D(
        drift=lambda x: kappa * (theta - x),
        diffusion_coeff=lambda _x: sigma,
        fd_step=fd_step,
    )


def black_scholes_adjoint_1d(
    mu: float, sigma: float, *, fd_step: float = 1e-4
) -> ClosedFormDiffusionAdjoint1D:
    """Black-Scholes example: dX = mu X dt + sigma X dW."""
    return ClosedFormDiffusionAdjoint1D(
        drift=lambda x: mu * x,
        diffusion_coeff=lambda x: sigma * x,
        fd_step=fd_step,
    )


@dataclass(slots=True)
class FiniteStateCTMCAdjoint(Generic[X], AdjointGenerator[DensityVector, Any]):
    """Adjoint generator for a finite-state CTMC with rate matrix Q."""

    states: Sequence[X]
    rate_matrix: Sequence[Sequence[float]]

    _index: dict[X, int] = field(init=False, repr=False)
    _rates: list[list[float]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n = len(self.states)
        if n == 0:
            raise ValueError("states must be nonempty")
        if any(len(row) != n for row in self.rate_matrix):
            raise ValueError("rate_matrix must be square (n x n)")
        self._index = {state: i for i, state in enumerate(self.states)}
        self._rates = [[float(v) for v in row] for row in self.rate_matrix]
        for i in range(n):
            row_sum = sum(self._rates[i][j] for j in range(n) if j != i)
            if abs(self._rates[i][i] + row_sum) > 1e-9:
                self._rates[i][i] = -row_sum

    def apply_to_density(self, p: DensityVector) -> list[float]:
        if len(p) != len(self.states):
            raise ValueError("density length must match states")
        n = len(self.states)
        adjoint = [0.0 for _ in range(n)]
        for i in range(n):
            for j in range(n):
                adjoint[j] += p[i] * self._rates[i][j]
        return adjoint

    def apply_to_law(self, mu: Any) -> Any:
        raise NotImplementedError(
            "FiniteStateCTMCAdjoint expects an explicit density vector."
        )


def _normalize_density_vector(p: DensityVector) -> list[float]:
    total = float(sum(p))
    if total <= 0:
        raise ValueError("density must have positive total mass")
    return [float(pi) / total for pi in p]


@dataclass(slots=True)
class StationaryDistributionSolver(Generic[X]):
    """Solve A^{*} p_{\infty} = 0 by averaging forward density evolution."""

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

    def solve_truncated(
        self, p0: DensityVector, *, lam: float, n_steps: int
    ) -> list[float]:
        """Truncated geometric-weighted average of densities."""
        if not (0.0 < lam < 1.0):
            raise ValueError("lam must be in (0, 1)")
        if n_steps <= 0:
            raise ValueError("n_steps must be > 0")
        if len(p0) != len(self.states):
            raise ValueError("initial density length must match states")
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
        rng: random.Random | None = None,
        seed: int | None = None,
    ) -> list[float]:
        """Unbiased geometric-horizon estimator for weighted infinite sum."""
        if not (0.0 < lam < 1.0):
            raise ValueError("lam must be in (0, 1)")
        if len(p0) != len(self.states):
            raise ValueError("initial density length must match states")
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