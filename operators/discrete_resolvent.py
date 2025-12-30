from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Hashable, MutableMapping, Optional, TypeVar
import random

from core_interfaces import MarkovKernel
from .custom_types import KeyFn, Observable, Scalar

X = TypeVar("X")


@dataclass(slots=True)
class DiscreteResolvent(Generic[X]):
    """
    Discrete resolvent for a Markov chain:

        U_\lambda f(x) = \Sum_{k=0}^\infty \lambda^k E_x[f(X_k)],    0<\lambda<1.

    Unbiased + fast Monte Carlo estimator of the discrete resolvent
    This version is unbiased:
    Let N be a random (geometric) stopping time defined by the rule:
        start at k=0
        while U < \lambda (U \sim Uniform[0,1] independent), continue to next step
        otherwise stop

    Then P(N >= k) = \lambda^k. Now consider the unweighted random-horizon sum
        Y = \Sum_{k=0}^N f(X_k).

    Taking expectation and using independence of N and the chain:

        E[Y]
        = E[ \Sum_{k=0}^\infty f(X_k) * 1_{N >= k} ]
        = \Sum_{k=0}^\infty E[f(X_k)] * P(N >= k)
        = \Sum_{k=0}^\infty E[f(X_k)] * \lambda^k
        = U_\lambda f(x0).

    So each path contribution Y is an unbiased estimator of U_\lambda f(x0), and
    averaging over `n_paths` preserves this unbiasedness.


    This implementation is faster compared to a naive implementation of U_\lambda via a weighted infinite series:
        \Sum_{k>=0} λ^k f(X_k),

    this is faster because:
      - It avoids computing \lambda^k (no exponentiation, no iterative weights).
      - It stops automatically at a random finite horizon N, with
            E[N] = \lambda / (1 - \lambda),
        so the expected number of kernel transitions per path is finite and controlled
        by \lambda.
      - The loop is tight: one Uniform draw + (maybe) one kernel sample per step.

    Practical note:
      - For \lambda close to 1, E[N] grows like 1/(1-\lambda), so runtime increases.
      - This type of fast estimator is typically used with bounded (or integrable) f.
    """

    kernel: MarkovKernel[X]
    lam: float
    method: str = "mc"

    # Caching hooks (same idea as semigroup):
    key_fn: Optional[KeyFn[X]] = None
    cache: Optional[MutableMapping[tuple, float]] = None

    def __post_init__(self) -> None:
        if not (0.0 < self.lam < 1.0):
            raise ValueError("lambda must be in (0,1)")

    def estimate_U(
        self,
        f: Observable[X],
        x0: X,
        *,
        n_paths: int,
        rng: Optional[random.Random] = None,
        seed: Optional[int] = None,
        f_key: Optional[Hashable] = None,
    ) -> Scalar:
        """
        Monte Carlo estimate of U_λ f(x0) using the unbiased geometric-horizon estimator.

        Monte Carlo estimator used here (unbiased, random horizon):
        Let N be geometric via: start k=0 and "continue" with probability λ each step.
        Then P(N ≥ k) = λ^k and

            E[ Σ_{k=0}^N f(X_k) ] = U_λ f(x0).

        This avoids explicit λ^k weighting along the path.

        Caching:
          - Only used when (cache provided) AND (key_fn provided) AND (seed provided).
          - Cache key includes: ("U", lam, n_paths, seed, x_key, f_key)
        """
        if n_paths <= 0:
            raise ValueError("n_paths must be > 0")
        if (rng is None) == (seed is None):
            raise ValueError("Provide exactly one of rng or seed")

        if self.cache is not None and self.key_fn is not None and seed is not None:
            xk = self.key_fn(x0)
            fk = f_key if f_key is not None else id(f)
            ck = ("U", self.lam, n_paths, seed, xk, fk)
            if ck in self.cache:
                return self.cache[ck]
        else:
            ck = None

        if rng is None:
            rng = random.Random(seed)

        acc: Scalar = 0.0
        for _ in range(n_paths):
            x = x0
            total = f(x)  # k=0
            while rng.random() < self.lam:
                x = self.kernel.law(x).sample(rng)
                total += f(x)  # unweighted sum => unbiased for U_λ
            acc += total

        est: Scalar = acc / n_paths
        if ck is not None:
            self.cache[ck] = est
        return est

    def estimate_U_truncated(
        self,
        f: Observable[X],
        x0: X,
        *,
        K: int,
        n_paths: int,
        rng: Optional[random.Random] = None,
        seed: Optional[int] = None,
    ) -> Scalar:
        """Estimate truncated series Σ_{k=0}^K λ^k f(X_k) by path simulation.

        This is *biased* (missing tail), but can be useful when you want explicit
        control of truncation (e.g. compare K→∞).
        """
        if K < 0:
            raise ValueError("K must be >= 0")
        if n_paths <= 0:
            raise ValueError("n_paths must be > 0")
        if (rng is None) == (seed is None):
            raise ValueError("Provide exactly one of rng or seed")
        if rng is None:
            rng = random.Random(seed)

        acc: Scalar = 0.0
        for _ in range(n_paths):
            x = x0
            total: Scalar = 0.0
            w = 1.0
            total += w * f(x)
            for _k in range(K):
                x = self.kernel.law(x).sample(rng)
                w *= self.lam
                total += w * f(x)
            acc += total
        return acc / n_paths

    def truncation_bias_bound(self, *, f_sup: float, K: int) -> float:
        """Deterministic sup-norm bound on the tail if |f|≤f_sup.

        |Σ_{k>K} λ^k T^k f| ≤ f_sup * λ^{K+1} / (1-λ)
        """
        if f_sup < 0:
            raise ValueError("f_sup must be >= 0")
        if K < 0:
            raise ValueError("K must be >= 0")
        return float(f_sup * (self.lam ** (K + 1)) / (1.0 - self.lam))