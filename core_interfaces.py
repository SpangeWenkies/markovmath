from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, TypeVar, Generic, Callable, runtime_checkable, Iterable
import random
import math

X = TypeVar("X")   # element / point
E = TypeVar("E")   # event representation


@runtime_checkable
class Event(Protocol[X]):
    def __call__(self, x: X) -> bool: ...


@dataclass(frozen=True)
class PredicateEvent(Generic[X]):
    pred: Callable[[X], bool]
    def __call__(self, x: X) -> bool:
        return self.pred(x)


class Space(Protocol[X]):
    """Underlying set of points (plus whatever structure you choose)."""
    # usually no methods needed; it’s a tag/interface


class MetricSpace(Protocol[X]):
    def dist(self, a: X, b: X) -> float: ...


class MeasurableSpace(Protocol[X, E]):
    """(X, Σ) where Σ is represented by the type E."""
    def whole(self) -> E: ...
    def empty(self) -> E: ...


class Measure(Protocol[X, E]):
    """μ: Σ -> [0, +∞]."""
    def measure(self, event: E) -> float: ...


class Sampler(Protocol[X]):
    """Sampling capability."""
    def sample(self, rng: random.Random) -> X: ...


class ProbabilityMeasure(Measure[X, E], Sampler[X], Protocol[X, E]):
    """Probability measure: total mass 1 (by contract)."""
    # you can add helpers if you want

class MarkovKernel(Protocol[X]):
    """K(x, ·) returns a law for the next state given current x."""
    def law(self, x: X) -> Sampler[X]: ...


@dataclass
class MarkovProcess(Generic[X]):
    init: Sampler[X]
    kernel: MarkovKernel[X]

    def sample_path(self, n: int, rng: random.Random | None = None) -> list[X]:
        rng = rng or random.Random()
        path: list[X] = []
        x = self.init.sample(rng)
        path.append(x)
        for _ in range(1, n):
            x = self.kernel.law(x).sample(rng)
            path.append(x)
        return path

@dataclass(frozen=True)
class BorelSpace(Generic[X]):
    metric: MetricSpace[X]        # carries topology
    is_polish: bool = True        # contract/metadata

    def whole(self) -> PredicateEvent[X]:
        return PredicateEvent(lambda _x: True)

    def empty(self) -> PredicateEvent[X]:
        return PredicateEvent(lambda _x: False)

@dataclass(frozen=True)
class OpenBall(Generic[X]):
    center: X
    radius: float
    metric: MetricSpace[X]

    def __call__(self, x: X) -> bool:
        return self.metric.dist(x, self.center) < self.radius


@dataclass(frozen=True)
class Complement(Generic[X]):
    a: Event[X]
    def __call__(self, x: X) -> bool:
        return not self.a(x)

@dataclass(frozen=True)
class Union(Generic[X]):
    parts: tuple[Event[X], ...]
    def __call__(self, x: X) -> bool:
        return any(p(x) for p in self.parts)

@dataclass(frozen=True)
class Intersection(Generic[X]):
    parts: tuple[Event[X], ...]
    def __call__(self, x: X) -> bool:
        return all(p(x) for p in self.parts)

class RMetric(MetricSpace[float]):
    def dist(self, a: float, b: float) -> float:
        return abs(a - b)

@dataclass(frozen=True)
class Normal(Sampler[float]):
    mean: float
    std: float
    def sample(self, rng: random.Random) -> float:
        return rng.gauss(self.mean, self.std)

@dataclass(frozen=True)
class RandomWalkKernel(MarkovKernel[float]):
    step_std: float
    def law(self, x: float) -> Sampler[float]:
        return Normal(mean=x, std=self.step_std)
    
def estimate_prob(law: Sampler[X], event: Event[X], n: int, rng: random.Random) -> float:
    hits = 0
    for _ in range(n):
        if event(law.sample(rng)):
            hits += 1
    return hits / n

mp = MarkovProcess(init=Normal(0.0, 1.0), kernel=RandomWalkKernel(step_std=0.5))
path = mp.sample_path(5, rng=random.Random(0))
print(path)
