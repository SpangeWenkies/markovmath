from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, Hashable, Optional, Sequence, TypeAlias, TypeVar
import random

from helper_funcs import _as_float_seq, _dot

from .discrete_semigroup import DiscreteSemigroup
from .custom_types import GeneratorDomain, GeneratorSource, Observable, Scalar

X = TypeVar("X")


@dataclass(slots=True)
class SampledGenerator(Generic[X]):
    """(Discretized) generator based on a one-step kernel interpreted as step Δt.

    A_Δt f(x) := (T f(x) - f(x)) / Δt

    - In true continuous-time theory, A is defined as a limit as Δt→0.
    - In discrete time, setting Δt=1 gives the standard difference operator.

    - It can be seen as the instantaneous drift of the expected value of the statistic

    - We can link the generator to the resolvent by the resolvent R_\alpha by u=R_\alpha f that in continuous time solves the static equations:
        (\alpha I - A)R_\alpha f = f
    - In the above case A is the operator that gives the evolution of the markov process, i.e.,
        P_t = e^{tA},   where P_t is a continuous semigroup operator


    """

    semigroup: DiscreteSemigroup[X]
    dt: float = 1.0
    domain: Optional[GeneratorDomain[X]] = None
    source: GeneratorSource = field(default=GeneratorSource.SAMPLED, init=False)

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be > 0")

    def estimate_Af(
        self,
        f: Observable[X],
        x0: X,
        *,
        n_samples: int,
        rng: Optional[random.Random] = None,
        seed: Optional[int] = None,
        f_key: Optional[Hashable] = None,
    ) -> Scalar:
        """Estimate A f(x0) via (T f - f)/dt."""
        if self.domain is not None and f not in self.domain:
            raise ValueError(
                "f must belong to the (rich) class of nicely behaving functions."
            )
        Tf = self.semigroup.estimate_T(
            f, x0, n_samples=n_samples, rng=rng, seed=seed, f_key=f_key
        )
        return (Tf - f(x0)) / self.dt


@dataclass(slots=True)
class ClosedFormGenerator(Generic[X]):
    """Closed-form generator for diffusions or jump processes.

    Supports:
      - Diffusions via drift b(x) and diffusion matrix a(x) (covariance).
      - Jump generators via explicit jump rates.
      - Completely custom generator through `custom_generator`.
    """

    drift: Optional[Callable[[X], Sequence[float]]] = None
    diffusion: Optional[Callable[[X], Sequence[Sequence[float]]]] = None
    jump_rates: Optional[Callable[[X], Sequence[tuple[X, float]]]] = None
    custom_generator: Optional[Callable[[Observable[X], X], Scalar]] = None
    domain: Optional[GeneratorDomain[X]] = None
    fd_step: float = 1e-4
    grad_fn: Optional[Callable[[Observable[X], X], Sequence[float]]] = None
    hess_fn: Optional[Callable[[Observable[X], X], Sequence[Sequence[float]]]] = None
    source: GeneratorSource = field(default=GeneratorSource.CLOSED_FORM, init=False)

    def __post_init__(self) -> None:
        if self.fd_step <= 0:
            raise ValueError("fd_step must be > 0")
        if (
            self.custom_generator is None
            and self.jump_rates is None
            and self.drift is None
            and self.diffusion is None
        ):
            raise ValueError(
                "Provide custom_generator, jump_rates, or drift/diffusion."
            )
        if self.jump_rates is not None and (
            self.drift is not None or self.diffusion is not None
        ):
            raise ValueError(
                "Jump generators must be specified without drift/diffusion."
            )

    def estimate_Af(self, f: Observable[X], x0: X) -> Scalar:
        if self.domain is not None and f not in self.domain:
            raise ValueError(
                "f must belong to the (rich) class of nicely behaving functions."
            )
        if self.custom_generator is not None:
            return self.custom_generator(f, x0)
        if self.jump_rates is not None:
            return self._jump_generator(f, x0)
        return self._diffusion_generator(f, x0)

    def _jump_generator(self, f: Observable[X], x0: X) -> Scalar:
        rates = self.jump_rates(x0) if self.jump_rates is not None else ()
        fx = f(x0)
        total: Scalar = 0.0
        for y, rate in rates:
            total += rate * (f(y) - fx)
        return total

    def _diffusion_generator(self, f: Observable[X], x0: X) -> Scalar:
        if self.drift is None and self.diffusion is None:
            raise ValueError("Diffusion generator requires drift or diffusion.")
        drift = self.drift(x0) if self.drift is not None else ()
        diffusion = self.diffusion(x0) if self.diffusion is not None else ()
        grad = self._grad(f, x0)
        hess = self._hess(f, x0)
        drift_term = _dot(_as_float_seq(drift), _as_float_seq(grad)) if drift else 0.0
        diffusion_term = 0.0
        if diffusion:
            diffusion_term = 0.5 * self._trace_product(diffusion, hess)
        return drift_term + diffusion_term

    def _grad(self, f: Observable[X], x0: X) -> Sequence[float]:
        if self.grad_fn is not None:
            return self.grad_fn(f, x0)
        return self._finite_diff_grad(f, x0)

    def _hess(self, f: Observable[X], x0: X) -> Sequence[Sequence[float]]:
        if self.hess_fn is not None:
            return self.hess_fn(f, x0)
        return self._finite_diff_hessian(f, x0)

    def _finite_diff_grad(self, f: Observable[X], x0: X) -> list[float]:
        x = _as_float_seq(x0)
        h = self.fd_step
        grad: list[float] = []
        for i in range(len(x)):
            xp = list(x)
            xm = list(x)
            xp[i] += h
            xm[i] -= h
            grad.append((f(tuple(xp)) - f(tuple(xm))) / (2.0 * h))
        return grad

    def _finite_diff_hessian(self, f: Observable[X], x0: X) -> list[list[float]]:
        x = _as_float_seq(x0)
        h = self.fd_step
        d = len(x)
        hess = [[0.0 for _ in range(d)] for _ in range(d)]
        fx = f(tuple(x))
        for i in range(d):
            xp = list(x)
            xm = list(x)
            xp[i] += h
            xm[i] -= h
            fpp = f(tuple(xp))
            fmm = f(tuple(xm))
            hess[i][i] = (fpp - 2.0 * fx + fmm) / (h**2)
            for j in range(i + 1, d):
                xpp = list(x)
                xpm = list(x)
                xmp = list(x)
                xmm = list(x)
                xpp[i] += h
                xpp[j] += h
                xpm[i] += h
                xpm[j] -= h
                xmp[i] -= h
                xmp[j] += h
                xmm[i] -= h
                xmm[j] -= h
                val = (
                    f(tuple(xpp)) - f(tuple(xpm)) - f(tuple(xmp)) + f(tuple(xmm))
                ) / (4.0 * h**2)
                hess[i][j] = val
                hess[j][i] = val
        return hess

    @staticmethod
    def _trace_product(
        a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]
    ) -> float:
        if len(a) != len(b):
            raise ValueError("dimension mismatch in trace product")
        total = 0.0
        for i, row in enumerate(a):
            if len(row) != len(b[i]):
                raise ValueError("dimension mismatch in trace product")
            for j, aij in enumerate(row):
                total += aij * b[i][j]
        return float(total)


Generator: TypeAlias = SampledGenerator[X] | ClosedFormGenerator[X]