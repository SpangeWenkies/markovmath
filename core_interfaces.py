from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, TypeVar, Generic, Callable, runtime_checkable, Iterable
import random
import math
import seaborn as sns

X = TypeVar("X")   # element / point
E = TypeVar("E")   # event representation

# TODO: create a contract testing file holding axiom checks by runtime / property-based tests

# we use a Protocol to be a contract describing required methods on a class

# How one can deal with axioms involving \forall or \exists in formal mathematics put into code is one of the following 4 ways:
    # Restricted representations where laws are checkable
        # e.g. finite state spaces: events are bitsets; probabilities are sums.
    # Runtime tests / property-based tests
        # e.g. Check triangle inequality for random samples.
    # Type-level discipline + documentation
        # “This must satisfy X.”
    # External proof assistants (Lean/Coq) if you want actual proof-level enforcement.

@runtime_checkable  # needed for working with isinstance() with protocols
class Event(Protocol[X]):
    def __call__(self, x: X) -> bool: ...   # ellipsis literal to state the method is not implimented but signature matters here


@dataclass(frozen=True)
class PredicateEvent(Generic[X]):
    # if event is e.g. (0, \infty) then its predicate is the function lambda x: x > 0
    # this function then wraps it so it can become a reusable "event object"
    pred: Callable[[X], bool]
    def __call__(self, x: X) -> bool:
        return self.pred(x)


class Space(Protocol[X]):
    """Underlying set of points (plus whatever structure you choose)."""
    # usually no methods needed; it’s a tag/interface


class MetricSpace(Protocol[X]):
    # defined by a contract, as we can not force metric axioms for all possible a,b
    # contract is s.t. user must promise to deliver a correct metric
    # python can only force that the dist method exists and its correct arguments must be delivered
    # TODO: insert metric axioms in this comment or inside some error handling
    def dist(self, a: X, b: X) -> float: ...


class MeasurableSpace(Protocol[X, E]):
    """(X, Σ) where Σ is represented by the type E."""
    # we cannot implement the closure of a sigma algebra as it may have infinitely many elements, we can define the following:
    def whole(self) -> E: ...   # produce event for the whole space
    def empty(self) -> E: ...   # produce event for the empty space


class Measure(Protocol[X, E]):
    """μ: Σ -> [0, +∞]."""
    # A measure must be countably additive. You cannot enforce “for every countable disjoint sequence of sets”.
    # TODO: insert measure axioms in comment or error message
    def measure(self, event: E) -> float: ...   # here is the function for the actual mapping the measure does and must be non-neg. target


class Sampler(Protocol[X]):
    """Sampling capability."""
    # generate random samples of type X
    def sample(self, rng: random.Random) -> X: ...


class ProbabilityMeasure(Measure[X, E], Sampler[X], Protocol[X, E]):
    """Probability measure: total mass 1 (by contract)."""
    # in general, many probability measures you can sample from don’t let you compute exact measure(event) for arbitrary events; 
    # this protocol assumes you can do both, but you could split it if needed.
    # helpers can be added here

class MarkovKernel(Protocol[X]):
    """K(x, ·) returns an evolution law for the next state given current x."""
    def law(self, x: X) -> Sampler[X]: ...


@dataclass
class MarkovProcess(Generic[X]):
    init: Sampler[X]    # initial distribution
    kernel: MarkovKernel[X] # transition kernel that has an evolution law function

    def sample_path(self, n: int, rng: random.Random | None = None) -> list[X]:
        rng = rng or random.Random() #If rng not provided, create a new RNG (non-reproducible unless seed is set)
        path: list[X] = []
        x = self.init.sample(rng)
        path.append(x)
        for _ in range(1, n):
            x = self.kernel.law(x).sample(rng)
            path.append(x)
        return path

@dataclass(frozen=True)
class StdBorelSpace(Generic[X]):
    
    metric: MetricSpace[X]        # induces standard topology
    is_polish: bool        # contract/metadata

    # This chooses the event representation E to be PredicateEvent[X].
    # So “events” are simply predicates.
    # This is pragmatic, but it’s not “all Borel sets”, it’s “whatever predicates you claim are measurable”.

    # You cannot represent “all Borel sets” in general because it involves countable operations and is too large.
    # So if you represent events by predicates, you’re relying on a contract:
    # “the predicate corresponds to a measurable set”

    def whole(self) -> PredicateEvent[X]:
        return PredicateEvent(lambda _x: True)

    def empty(self) -> PredicateEvent[X]:
        return PredicateEvent(lambda _x: False)

@dataclass(frozen=True)
class OpenBall(Generic[X]):
    # represents open ball with center and radius on a space equiped with some metric defined by the metric space
    center: X
    radius: float
    metric: MetricSpace[X]

    # callable returns if x is member of the open ball
    def __call__(self, x: X) -> bool:
        return self.metric.dist(x, self.center) < self.radius


@dataclass(frozen=True)
class Complement(Generic[X]):
    a: Event[X]
    def __call__(self, x: X) -> bool:
        return not self.a(x)

# finite union and finite intersection together with the open balls can create a computable part of the sigma-algebra
# note that we can't do countable as that can reach infinity

@dataclass(frozen=True)
class Union(Generic[X]):
    # finite union
    parts: tuple[Event[X], ...]
    def __call__(self, x: X) -> bool:
        return any(p(x) for p in self.parts)

@dataclass(frozen=True)
class Intersection(Generic[X]):
    # finite intersection
    parts: tuple[Event[X], ...]
    def __call__(self, x: X) -> bool:
        return all(p(x) for p in self.parts)


class RMetric(MetricSpace[float]):
    # makes it a R^1 metric space and implements the euclidian distance function
    def dist(self, a: float, b: float) -> float:
        return abs(a - b)

@dataclass(frozen=True)
class Normal(Sampler[float]):
    # makes the sampler a gaussian sampler
    mean: float
    std: float
    def sample(self, rng: random.Random) -> float:
        return rng.gauss(self.mean, self.std)

@dataclass(frozen=True)
class RandomWalkKernel(MarkovKernel[float]):
    # makes a transition kernel one that uses a gaussian random walk
    # given current state x, the next state is normal(x,step_std)
    step_std: float
    def law(self, x: float) -> Sampler[float]:
        return Normal(mean=x, std=self.step_std)
    
def estimate_prob(law: Sampler[X], event: Event[X], n: int, rng: random.Random) -> float:
    # Draws n samples from law.
    # Counts how many fall in the event.
    # Returns the empirical frequency, an estimate of P(X \in event)
    hits = 0
    for _ in range(n):
        if event(law.sample(rng)):
            hits += 1
    return hits / n

mp = MarkovProcess(init=Normal(0.0, 1.0), kernel=RandomWalkKernel(step_std=0.5))
path = mp.sample_path(100, rng=random.Random(0))
sns.lineplot(path)
