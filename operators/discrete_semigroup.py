from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Hashable, MutableMapping, Optional, TypeVar
import random

from core_interfaces import MarkovKernel
from .custom_types import KeyFn, Observable, Scalar

X = TypeVar("X")


@dataclass(slots=True)
class DiscreteSemigroup(Generic[X]):
    """
    Discrete-time semigroup associated to a (time-homogeneous) Markov kernel K:

        (T f)(x)    = E_x[f(X_1)]
        (T^n f)(x)  = E_x[f(X_n)]

    This class provides Monte Carlo estimators of T^n f(x).
    """

    kernel: MarkovKernel[X]

    # Caching hooks:
    # - If your state X is not hashable (common in R^d), provide a key_fn,
    #   e.g. rounding coordinates to a grid.
    key_fn: Optional[KeyFn[X]] = None
    cache: Optional[MutableMapping[tuple, float]] = None

    method: str = "mc"

    def estimate_Tn(
        self,
        f: Observable[X],
        x0: X,
        *,
        n: int,
        n_samples: int,
        rng: Optional[random.Random] = None,
        seed: Optional[int] = None,
        f_key: Optional[Hashable] = None,
    ) -> Scalar:
        """
        Monte Carlo estimate of T^n f(x0) = E[f(X_n) | X_0=x0].

        Caching:
          - Only used when (cache is provided) AND (key_fn is provided) AND (seed is provided).
          - Cache key includes: ( "Tn", n, n_samples, seed, x_key, f_key )
          - If you don't pass f_key, we default to id(f), which is stable only within a process.
        """
        if n < 0:
            raise ValueError("n must be >= 0")
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        if (rng is None) == (seed is None):
            raise ValueError("Provide exactly one of rng or seed")

        if self.cache is not None and self.key_fn is not None and seed is not None:
            xk = self.key_fn(x0)
            fk = f_key if f_key is not None else id(f)
            ck = ("Tn", n, n_samples, seed, xk, fk)
            if ck in self.cache:
                return self.cache[ck]
        else:
            ck = None

        if rng is None:
            rng = random.Random(seed)

        acc: Scalar = 0.0
        for _ in range(n_samples):
            x = x0
            for _ in range(n):
                x = self.kernel.law(x).sample(rng)
            acc += f(x)
        est: Scalar = acc / n_samples

        if ck is not None:
            self.cache[ck] = est
        return est

    def estimate_T(
        self,
        f: Observable[X],
        x0: X,
        *,
        n_samples: int,
        rng: Optional[random.Random] = None,
        seed: Optional[int] = None,
        f_key: Optional[Hashable] = None,
    ) -> Scalar:
        """Convenience wrapper for T^1."""
        return self.estimate_Tn(
            f, x0, n=1, n_samples=n_samples, rng=rng, seed=seed, f_key=f_key
        )