# Needed for semigroups & resolvents

# Say our measurable space is (S,\Sigma)

# In our framework we do not have the transition function K(x,A): \Sigma -> [0,1] explicitly
# We use a sampler kernel.law(x) to return draws from the K(x,.)

# If we have a bounded function f: S -> R^1 we define the Markov operator Tf as
# (Tf)(x) = \int_S f(y)K(x,dy) = E[f(X_1)|X_0=x]
# this markov operator is used in the unfinished contract checker aswell

# A discrete semigroup gives (T^n f)(x) = T^n f(x) = E[f(X_n)|X_0=x]
# i.e. it satisfies the semigroup property (discrete Chapman-Kolmogorov for time-homogeneous markov process)
# T^{m+n} = T^m after T^n

# Note: if we have A \in \Sigma and take f(x) = 1_A(x) then we get the following
# T^n f(x) = P_x[X_n \in A]
# generated Borel-ish event family then becomes a library of test functions via indicators functions

# Resolvent (which is in our case the discounted potential operator) we can define as follows:
# for 0<\lambda<1 and a function f bounded and measurable
# (U_{\lambda} f)(x) = \Sum_{k=0}^{\infty} { \lambda^k (P^k f)(x) } = E[\Sum_{k=0}^{\infty} { \lambda^k f(X_k)}]
# This is the discrete time analogue of the continuous time resolvent (U_\alpha)_{\alpha>0}
# We get some key identities from this:
# U_\lambda f = f + \lambda P(U_\lambda f)
# equivalent to:    (I - P)U_\lambda f = f

# Note: if we have A \in \Sigma and take f(x) = 1_A(x) then we can see (U_\lambda 1_A)(x) as
# an expected discounted occupation time of set A starting from x

# To approximate an infinite sum we can truncate the sum
# We then need to evaluate the truncation errors to prove convergence and look at the rate of convergence
# There are multiple ways to look at these depending on the series properties:
# If we have alternating series test being satisfied we get an exact error evaluation
# If we have a hypergeometric series, we can use the ratio to get the evaluation of the error term
# For a matrix exponential (which is an infinite series), there is a scaling and squaring method error evaluation
# are there more???
# If we have bounded f then |f|<=M, then also |T^k f| <= M (intuitively this is as Markov operators are contractions in sup norm)
# if Bias(K) is the infinite sum minus the truncated sum (up to K) then:
# |Bias(K)| <= \Sum^{\infty}_{k=K+1} \lambda^k M
# the latter is a geometric sum as 0<\lambda<1
# thus |Bias(K)| <= M (\lambda^{K+1}) / (\lambda - 1)

# TODO 27/12/2025 for semigroup and resolvent operator layer:
# we must look at more test functions that are usefull and what comes after getting the semigroups and resolvents from these, e.g.,
# oscillator function sin(x) or e^{i \xi x} usefull for fourier and spectral methods
# moment functions
# "energy"/growth functions like the squared norm ||.||^2
# mean type functions like f(x)=x and f(x)=<v,x>
# create some payoff functions f s.t. semigroup is a value function and the resolvent is the discounted total value
# calculating the generator A such that we can use PDE and martingale tools, which way around? can we use P_t = e^{tA} or use limit?
# create continuous time versions
# do we use montecarlo integration?
# after we have generator A a function that finds the b and c in stability formula Af <= -cf + b
# Kolmogorov backward and forward???????
# can we find resolvent from A or other way around using static equations (\alpha I - A)R_{\alpha}f = f
# what can we do with that when we know A on a "rich" class of f then we get a function in A and f that is a Martingale
# note somewhere that we need for nice expectations that f are bounded, cont., smooth, of compact support
# note also then properties we want for P_t like positivity preservation, contraction, strong cont., invariant measure, Feller, Spectral gap
# for resolvents
# in cont time make stopping time tau \ sim Exp(\alpha) random stopping time
# in dicrete time make \lambda s.t. sum is truncated at geometric stopping time with probability of stopping each step being /lambda
# can we check chapman-kolmogorov heuristicly for a P_t?

# TODO: 29/12/2025 implementations and questions:
# explore implication of link from markov chains to martingales again, does it only then mean we can speak of the theory below?
# if we know A on a "rich" class of f then we get a function in A and f that is a Martingale, what is this knowing on a rich class
# what stability is found when we have Af <= -cf + b as opposed to stability of distribution spoken of below?
# implement QM methods for solving Fokker-Planck by using analog to schrödinger's equation
# create a trigger that sees when FP eq only shows overdamped dynamics, i.e., only 2nd order partials w.r.t. spatial variables
# we should then be able to write down a master equation and easily solve it numerically
# find out if continuous time wrapper for the semigroups and resolvent is even sensible or if it is like MC for discretized PDE case
# clean forward and backword kolmogorov framework with fokker-planck in forward case if we have a density
# forward eq can either be with a law /mu_t or a density p, what do we do with this? is law programming sensible
# how to implement PDE's then SDE's then also diffusion equations
# make the demo into a demo for payoff of EU puts
# add solvers for PDE's: Euler-Marumaya, Milstein, Runge-Kufta, Rosenbrock, more (see Kuznetsov 2023 and Rybakov 2023)
# solvers for PDE being Stochastic with Lebesgue integral, Itô integral, (Lévy integral)
# make generator A be possibly a closed form generator (now it is only made from sampling i think)
# how to ensure we keep the bond between the Markov chains and the stochestic processes
# do we want to do something with weak vs strong sol. of a SDE (Yamada-Watanabe theorem)
# make the distinction that the MC simulation of average paths to calc semigroup is canonical ensembling
# make the distinction that the other way to calc the semigroup or resolvent is thus through closed form generator and SDE solving
# think of when what calculation is best, canonical ensembling, PDE analytical or even MC in discretized PDE
# create the framework that will find the local volatility (problem of inversion of the Fokker-Planck eqation)
# non parametrically: Dupire
# parametrically: Brigo & Mercurio (2003)
# for more info see Fengler (2008), Gatheral (2008), Musiela & Rutkowski (2008)
# think of when A can not be written closed form, which black-box transitions, learned kernels, complicated algorithmic dynamics are common
# maybe add these as demo's
# When we do have a closed form generator we might want to have some example closed forms
# corresponding to e.g. Black-Scholes, OU, Heston, CIR, etc
# in general these are Af(x) = b(x) * laplace f(x) + 1/2 \trace (a(x) laplace^2 f(x)), where a(x) = \sigma \sigma' (for smooth f)
# we also have closed forms if comming from continuous time Markov chain where we know the jump rates. This might be the main link!!!
# we have closed forms if we have Lévy / jump diffusions (maybe make examples of these aswell)
# Create a way to calculate the adjoint either numerically or analytically
# analytically we can use integration by parts
# numerically we can use CTMC
# analytically there should be more techniques in lecture on quantum mechanics (also on existence of adjoint and self adjointness etc.)
# providing closed form of adjoint should also be possible just like with the generator
# we will use the adjoint generator for the following things
# evolving either a law or density forward in time with the forward kolmogorov / Fokker-Planck equations
# computing transition densities / likelihoods for calibration / inference of something (for what?)
# finding stationary / invariant distributions by solving A^{*}p_{\infty} = 0 with normalization
# do a forward pricing trick: compute p(t,.|x) and price many payoffs at once by integrating u(t,x) = \int f(y)p(t,y|x)dy
# use it in risk setting by looking at the result of the forward equations for use in finding tail risk, (hitting probability)
# consider discounting in both the semigroup (to get Feynman-Kac) and in the resolvent (where it will just shift alpha)
# make the discounting possibly state-dependent
# Fokker-Planck equation path integral implementation
# I want to create a file that can show all kinds of (R^1 to R^3) demo's as graphs of things


from dataclasses import dataclass, field
from typing import (
    Callable,
    TypeVar,
    Hashable,
    Generic,
    Optional,
    MutableMapping,
    Sequence,
    Protocol,
    Iterator,
    runtime_checkable,
    Any,
    TypeAlias,
)
from enum import Enum
import random
from core_interfaces import (
    MarkovKernel,
    ProbabilityMeasure,
    LawEvolution,
    DensityEvolution,
    Density,
)
from helper_funcs import (
    _as_float_seq,
    _dot,
)
import cmath
import math

X = TypeVar("X")
E = TypeVar("E")

Observable = Callable[
    [X], float
]  # we define now an observable as we can not directly observe events but we do observe an operator/map

Scalar = float | complex  # to let functions return either float or complex number

DensityVector: TypeAlias = Sequence[float]

# For R^d (general state space), exact caching by state almost never hits because each sample is new.
# The caching hook is still useful if you supply a coarse key_fn, e.g. round to a grid by latter defined rd_key
# The caching will speed up repeated evaluation during the contract checks
KeyFn = Callable[[X], Hashable]

class GeneratorSource(str, Enum):
    CLOSED_FORM = "closed_form"
    SAMPLED = "sampled"

# if we know A on a rich class of test functions then we can link the notion of a martingale to the markov chain
# to concretize this we define a domain of test functions for which we know A below
# A "domain" is thus a family of test functions together with assumptions
# (e.g., smoothness, boundedness) used when the generator is defined.
@runtime_checkable
class GeneratorDomain(Protocol[X]):
    functions: Sequence[Observable[X]]
    assumptions: Sequence[str]

    def __iter__(self) -> Iterator[Observable[X]]: ...
    def __contains__(self, f: object) -> bool: ...


@dataclass(frozen=True, slots=True)
class FiniteGeneratorDomain(Generic[X]):
    """Concrete container for a finite rich class of test functions."""

    functions: Sequence[Observable[X]]
    assumptions: Sequence[str] = ()

    def __iter__(self) -> Iterator[Observable[X]]:
        return iter(self.functions)

    def __contains__(self, f: object) -> bool:
        return f in self.functions


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


# p = DiscreteSemigroup.estimate_Tn(indicator(A), x0, n=10, n_samples=5000, rng=rng, seed=seed, f_key=f_key)
# gives the probability P(X_10 ∈ A | x0)


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


# -----------------------------
# Generator (sampled / closed-form)
# -----------------------------


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
    source: GeneratorSource = field(
        default=GeneratorSource.SAMPLED, init=False
    )

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
    source: GeneratorSource = field(
        default=GeneratorSource.CLOSED_FORM, init=False
    )

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
                    f(tuple(xpp))
                    - f(tuple(xpm))
                    - f(tuple(xmp))
                    + f(tuple(xmm))
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

# -----------------------------
# Adjoint generators
# -----------------------------

DensityLike = TypeVar("DensityLike")
LawLike = TypeVar("LawLike")


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
        rng: Optional[random.Random] = None,
        seed: Optional[int] = None,
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

# -----------------------------
# Continuous-time (discretized) wrappers
# -----------------------------

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

@dataclass(slots=True)
class ForwardEquation(Generic[X, E]):
    """
    Forward Kolmogorov/Fokker-Planck wrapper that dispatches on representation.

    Provide either a law solver or a density solver (or both).
    """

    law_solver: Optional[LawEvolution[X, E]] = None
    density_solver: Optional[DensityEvolution[X]] = None

    def forward_law_step(self, mu0: ProbabilityMeasure[X, E], t: float) -> ProbabilityMeasure[X, E]:
        if self.law_solver is None:
            raise ValueError("No law solver configured for ForwardEquation.")
        return self.law_solver.evolve_law(mu0, t)

    def forward_density_step(self, p0: Density[X], t: float) -> Density[X]:
        if self.density_solver is None:
            raise ValueError("No density solver configured for ForwardEquation.")
        return self.density_solver.evolve_density(p0, t)

    def forward(self, initial: ProbabilityMeasure[X, E] | Density[X], t: float) -> ProbabilityMeasure[X, E] | Density[X]:
        if self._is_law(initial):
            return self.forward_law_step(initial, t)
        return self.forward_density_step(initial, t)

    @staticmethod
    def _is_law(obj: Any) -> bool:
        return hasattr(obj, "sample") or hasattr(obj, "measure")

# -----------------------------
# Test functions with wrappers for R^d
# -----------------------------


def coordinate(i: int) -> Observable[tuple[float, ...]]:
    """f(x)=x_i"""

    def f(x: tuple[float, ...]) -> float:
        return float(x[i])

    return f


def linear(v: Sequence[float]) -> Observable[tuple[float, ...]]:
    """f(x)=<v,x>"""
    v_ = [float(vi) for vi in v]

    def f(x: tuple[float, ...]) -> float:
        return _dot(v_, _as_float_seq(x))

    return f


def squared_norm(p: float = 2.0) -> Observable[tuple[float, ...]]:
    """Energy/growth function: f(x)=||x||_p^2 (default p=2)."""
    if p <= 0:
        raise ValueError("p must be > 0")

    def f(x: tuple[float, ...]) -> float:
        xs = _as_float_seq(x)
        if p == 2.0:
            return float(sum(xi * xi for xi in xs))
        # ||x||_p = (Σ|xi|^p)^{1/p}, then squared
        norm_p = sum(abs(xi) ** p for xi in xs) ** (1.0 / p)
        return float(norm_p * norm_p)

    return f


def monomial(powers: Sequence[int]) -> Observable[tuple[float, ...]]:
    """Moment monomial: f(x)=∏_i x_i^{powers[i]} (powers are nonnegative ints)."""
    pw = list(powers)
    if any(k < 0 for k in pw):
        raise ValueError("powers must be nonnegative")

    def f(x: tuple[float, ...]) -> float:
        xs = _as_float_seq(x)
        if len(xs) != len(pw):
            raise ValueError("dimension mismatch: x vs powers")
        out = 1.0
        for xi, ki in zip(xs, pw):
            out *= xi**ki
        return float(out)

    return f


def sin_frequency(xi: Sequence[float]) -> Observable[tuple[float, ...]]:
    """Oscillator: f(x)=sin(<xi,x>). Useful for Fourier / spectral heuristics."""
    xi_ = [float(a) for a in xi]

    def f(x: tuple[float, ...]) -> float:
        return math.sin(_dot(xi_, _as_float_seq(x)))

    return f


def complex_exponential(xi: Sequence[float]) -> Observable[tuple[float, ...]]:
    """Complex exponential: f(x)=exp(i <xi,x>). (Returns complex.)"""
    xi_ = [float(a) for a in xi]

    def f(x: tuple[float, ...]) -> complex:
        return cmath.exp(1j * _dot(xi_, _as_float_seq(x)))

    return f


def payoff_call(strike: float, idx: int = 0) -> Observable[tuple[float, ...]]:
    """European call payoff: f(x)=max(x[idx]-K, 0)."""
    K = float(strike)

    def f(x: tuple[float, ...]) -> float:
        return max(float(x[idx]) - K, 0.0)

    return f


def payoff_put(strike: float, idx: int = 0) -> Observable[tuple[float, ...]]:
    """European put payoff: f(x)=max(K-x[idx], 0)."""
    K = float(strike)

    def f(x: tuple[float, ...]) -> float:
        return max(K - float(x[idx]), 0.0)

    return f
