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
    # for 0<=\lambda<=1 and a function f bounded and measurable
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

from typing import Callable, TypeVar
import random

X = TypeVar("X")
Observable = Callable[[X], float]   # we define now an observable as we can not directly observe events but we do observe an operator/map

def mc_Tn_f(
    kernel, # MarkovKernel[X]?
    x0: X,
    f: Observable[X],
    n: int,
    n_samples: int,
    rng: random.Random,
) -> float:
    """
    Monte Carlo estimator of T^n f(x0) = E[f(X_n) | X_0=x0].
    Contract here is that we have a bounded measurable function f:S -> R^1 (or domain being a subset of S?)
    """
    acc = 0.0
    for _ in range(n_samples):
        x = x0
        for _ in range(n):
            x = kernel.law(x).sample(rng)
        acc += f(x)
    return acc / n_samples

def indicator(A):
    return lambda x: 1.0 if A(x) else 0.0

# p = mc_Tn_f(kernel, x0, indicator(A), n=10, n_samples=5000, rng=rng) gives the probability P_x0(X_10 ∈ A)


