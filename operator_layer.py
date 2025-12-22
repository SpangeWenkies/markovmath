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
    
from dataclasses import dataclass    
from typing import Callable, TypeVar, Hashable, Generic, Optional, MutableMapping
import random
from core_interfaces import(
    MarkovKernel,
)

X = TypeVar("X")
Observable = Callable[[X], float]   # we define now an observable as we can not directly observe events but we do observe an operator/map
# For R^d (general state space), exact caching by state almost never hits because each sample is new. 
# The caching hook is still useful if you supply a coarse key_fn, e.g. round to a grid by latter defined rd_key
# The caching will speed up repeated evaluation during the contract checks
KeyFn = Callable[[X], Hashable]

def rd_key(x: tuple[float, ...], ndigits: int = 2) -> tuple:
    return tuple(round(xi, ndigits) for xi in x)

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
    ) -> float:
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

        acc = 0.0
        for _ in range(n_samples):
            x = x0
            for _ in range(n):
                x = self.kernel.law(x).sample(rng)
            acc += f(x)
        est = acc / n_samples

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
    ) -> float:
        """Convenience wrapper for T^1."""
        return self.estimate_Tn(
            f, x0, n=1, n_samples=n_samples, rng=rng, seed=seed, f_key=f_key
        )

def indicator(A: Callable[[X], bool]) -> Observable[X]:
    """Turn an event A(x)->bool into the indicator observable 1_A(x)."""
    return lambda x: 1.0 if A(x) else 0.0

# p = DiscreteSemigroup.estimate_Tn(indicator(A), x0, n=10, n_samples=5000, rng=rng, seed=seed, f_key=f_key) 
# gives the probability P(X_10 ∈ A | x0)

@dataclass(slots=True)
class DiscountedResolvent(Generic[X]):
    """
    Discounted (discrete-time) resolvent for a Markov chain:

        U_\lambda f(x) = \Sum_{k=0}^\infty \lambda^k E_x[f(X_k)],    0<\lambda<1.

    Unbiased + fast Monte Carlo estimator of the discounted resolvent
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

    # Caching hooks (same idea as semigroup):
    key_fn: Optional[KeyFn[X]] = None
    cache: Optional[MutableMapping[tuple, float]] = None

    def __post_init__(self) -> None:
        if not (0.0 < self.lam < 1.0):
            raise ValueError("lam must be in (0,1)")

    def estimate_U(
        self,
        f: Observable[X],
        x0: X,
        *,
        n_paths: int,
        rng: Optional[random.Random] = None,
        seed: Optional[int] = None,
        f_key: Optional[Hashable] = None,
    ) -> float:
        """
        Monte Carlo estimate of U_λ f(x0) using the unbiased geometric-horizon estimator.

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

        acc = 0.0
        for _ in range(n_paths):
            x = x0
            total = f(x)  # k=0
            while rng.random() < self.lam:
                x = self.kernel.law(x).sample(rng)
                total += f(x)  # unweighted sum => unbiased for U_λ
            acc += total

        est = acc / n_paths
        if ck is not None:
            self.cache[ck] = est
        return est

