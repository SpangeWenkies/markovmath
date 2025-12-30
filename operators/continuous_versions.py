from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Hashable, MutableMapping, Optional, TypeVar
import math
import random

from core_interfaces import MarkovKernel
from .discrete_resolvent import DiscreteResolvent
from .discrete_semigroup import DiscreteSemigroup
from .custom_types import KeyFn, Observable, Scalar

X = TypeVar("X")


@dataclass(slots=True)
class ContinuousSemigroup(Generic[X]):
    """Continuous-time semigroup built from a small-step kernel.

    Interpret the underlying kernel step as Δt, and approximate:

        P_t f(x) ≈ T^{n} f(x),  n = round(t/Δt)

    This is the typical situation for time-discretized SDEs.
    """

    kernel_dt: MarkovKernel[X]
    dt: float

    key_fn: Optional[KeyFn[X]] = None
    cache: Optional[MutableMapping[tuple, Scalar]] = None
    method: str = "mc"

    _disc: DiscreteSemigroup[X] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        self._disc = DiscreteSemigroup[X](
            kernel=self.kernel_dt,
            method=self.method,
            key_fn=self.key_fn,
            cache=self.cache,
        )

    def n_steps(self, t: float) -> int:
        if t < 0:
            raise ValueError("t must be >= 0")
        return int(round(t / self.dt))

    def estimate_Pt(
        self,
        f: Observable[X],
        x0: X,
        *,
        t: float,
        n_samples: int,
        rng: Optional[random.Random] = None,
        seed: Optional[int] = None,
        f_key: Optional[Hashable] = None,
    ) -> Scalar:
        n = self.n_steps(t)
        return self._disc.estimate_Tn(
            f, x0, n=n, n_samples=n_samples, rng=rng, seed=seed, f_key=f_key
        )


@dataclass(slots=True)
class ContinuousResolvent(Generic[X]):
    """Continuous-time resolvent built from a small-step kernel.

        R_\alpha f(x) = ∫_0^∞ e^{-\alpha t} P_t f(x) dt

    Discretization with step Δt:

        R_\alpha f(x) ≈ Δt Σ_{k=0}^∞ e^{-\alpha kΔt} (T^k f)(x)
               = Δt * U_λ f(x)   with  λ = e^{-\alpha Δt}.

    This implementation uses the unbiased geometric-stopping estimator for U_λ,
    and then multiplies by Δt.
    """

    kernel_dt: MarkovKernel[X]
    dt: float
    alpha: float

    key_fn: Optional[KeyFn[X]] = None
    cache: Optional[MutableMapping[tuple, Scalar]] = None

    _disc_res: DiscreteResolvent[X] = field(init=False, repr=False)

    method: str = "mc"

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0")
        lam = math.exp(-self.alpha * self.dt)
        self._disc_res = DiscreteResolvent[X](
            kernel=self.kernel_dt,
            lam=lam,
            method=self.method,
            key_fn=self.key_fn,
            cache=self.cache,
        )

    @property
    def lam(self) -> float:
        return float(self._disc_res.lam)

    def estimate_Ralpha(
        self,
        f: Observable[X],
        x0: X,
        *,
        n_paths: int,
        rng: Optional[random.Random] = None,
        seed: Optional[int] = None,
        f_key: Optional[Hashable] = None,
    ) -> Scalar:
        return self.dt * self._disc_res.estimate_U(
            f, x0, n_paths=n_paths, rng=rng, seed=seed, f_key=f_key
        )

    def estimate_Ralpha_via_exp_time(
        self,
        f: Observable[X],
        x0: X,
        *,
        n_paths: int,
        rng: Optional[random.Random] = None,
        seed: Optional[int] = None,
    ) -> Scalar:
        """Alternative estimator using τ ~ Exp(alpha) (approximate under time-discretization).

        Theory: R_\alpha f = (1/\alpha) E[f(X_τ)] for τ independent Exp(\alpha).

        Here we can only simulate at multiples of Δt, so we approximate X_τ by X_{ceil(τ/Δt)Δt}.
        """
        if n_paths <= 0:
            raise ValueError("n_paths must be > 0")
        if (rng is None) == (seed is None):
            raise ValueError("Provide exactly one of rng or seed")
        if rng is None:
            rng = random.Random(seed)

        acc: Scalar = 0.0
        for _ in range(n_paths):
            # sample Exp(alpha) via inverse CDF
            u = rng.random()
            tau = -math.log(max(u, 1e-15)) / self.alpha
            n = int(math.ceil(tau / self.dt))
            # evolve n steps (approx)
            x = x0
            for _k in range(n):
                x = self.kernel_dt.law(x).sample(rng)
            acc += f(x)
        return (acc / n_paths) / self.alpha