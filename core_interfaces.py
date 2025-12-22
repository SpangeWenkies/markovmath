from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, TypeVar, Generic, Callable, runtime_checkable, TypeAlias, Any
import random
import math

X = TypeVar("X")   # element / point
E = TypeVar("E")   # event representation

# in R^d do not use np.array as these are not hashable, do Point = tuple[float, ...] of length d
PointRd: TypeAlias = tuple[float, ...]

# TODO: finalize a contract testing file holding axiom checks by runtime / property-based tests

# TODO: would it be handy to use from typing the import Iterable

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
    # in our model an event is just anything callable
    # later on events can become structured events like Union((A, B)), Complement(A), OpenBall(center, radius, metric)
        # of those we can check what they are built from by methods .parts (or .a in complement case)
    # other events are opaque, i.e., they are "black-boxes" represented by (lambda) functions
        # these can be evaluated but not algebraically manipulated
    def __call__(self, x: X) -> bool: ...   # ellipsis literal to state the method is not implimented but signature matters here

@dataclass(frozen=True, slots=True)
class WholeEvent(Generic[X]):
    # represents the whole event set
    def __call__(self, x: X) -> bool:
        return True

@dataclass(frozen=True, slots=True)
class EmptyEvent(Generic[X]):
    # represents the empty event set
    def __call__(self, x: X) -> bool:
        return False

@dataclass(frozen=True, slots=True)
class NamedEvent(Generic[X]):
    """
    Wrap an arbitrary predicate with a stable identity (name).
    Helpful if you want simplification/dedup to treat two occurrences as the same event.
    """
    name: str
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
    # in general, many probability measures we can sample from don’t let us compute exact measure(event) for arbitrary events; 
    # this protocol assumes you can do both, but we could split it if needed.
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

@dataclass(frozen=True, slots=True)
class StdBorelSpaceRd(MeasurableSpace[PointRd, Event[PointRd]]):
    """
    A pragmatic representation of (R^d, B(R^d)).

    metric: the L^p metric chosen by the user (p-norm). On finite-dimensional R^d,
            any L^p norm induces the standard topology, hence the standard Borel σ-algebra.
    """
    metric: MetricSpace[PointRd]
    is_polish: bool = True  # contract/metadata

    def whole(self) -> Event[PointRd]:
        return WholeEvent()

    def empty(self) -> Event[PointRd]:
        return EmptyEvent()

    def ball(self, center: PointRd, radius: float) -> Event[PointRd]:
        return OpenBall(center=center, radius=radius, metric=self.metric)

@dataclass(frozen=True, slots=True)
class OpenBall(Generic[X]):
    # represents open ball with center and radius on a space equiped with some metric defined by the metric space
    center: X
    radius: float
    metric: MetricSpace[X]

    # callable returns if x is member of the open ball
    def __call__(self, x: X) -> bool:
        return self.metric.dist(x, self.center) < self.radius


@dataclass(frozen=True, slots=True)
class Complement(Generic[X]):
    a: Event[X]
    def __call__(self, x: X) -> bool:
        return not self.a(x)

# finite union and finite intersection together with the open balls can create a computable part of the sigma-algebra
# note that we can't do countable as that can reach infinity

@dataclass(frozen=True, slots=True)
class Union(Generic[X]):
    # finite union
    parts: tuple[Event[X], ...]
    def __call__(self, x: X) -> bool:
        return any(p(x) for p in self.parts)

@dataclass(frozen=True, slots=True)
class Intersection(Generic[X]):
    # finite intersection
    parts: tuple[Event[X], ...]
    def __call__(self, x: X) -> bool:
        return all(p(x) for p in self.parts)
    
# For now we only implement the Lp-norms as these introduce the standard topology on R^d
# TODO: extend to use other norms that induce other topologies

@dataclass(frozen=True, slots=True)
class LpMetricRd(MetricSpace[PointRd]):
    """
    L^p metric on R^d:
      - p >= 1: (sum |xi-yi|^p)^(1/p)
      - p = inf: max_i |xi-yi| (this is the supremum norm)
    """
    p: float  # use math.inf for p = infinity

    def dist(self, a: PointRd, b: PointRd) -> float:
        if len(a) != len(b):
            raise ValueError("LpMetricRd: points must have same dimension")
        if self.p == math.inf:
            return max(abs(ai - bi) for ai, bi in zip(a, b))
        if self.p < 1:
            raise ValueError("LpMetricRd: requires p >= 1 (or p = inf)")
        s = 0.0
        for ai, bi in zip(a, b):
            s += abs(ai - bi) ** self.p
        return s ** (1.0 / self.p)

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
    
@dataclass(frozen=True, slots=True)
class NormalRd(Sampler[PointRd]):
    """N(mean, std^2 I) on R^d (independent coordinates)."""
    mean: PointRd
    std: float

    def sample(self, rng: random.Random) -> PointRd:
        return tuple(rng.gauss(m, self.std) for m in self.mean)


@dataclass(frozen=True, slots=True)
class RandomWalkKernelRd(MarkovKernel[PointRd]):
    """X_{t+1} = X_t + N(0, step_std^2 I)."""
    step_std: float

    def law(self, x: PointRd) -> Sampler[PointRd]:
        return NormalRd(mean=x, std=self.step_std)

# ---------- Keys (for canonicalization/dedup in the AST (Abstract Syntax Tree)) ----------

def _round_float(v: float, ndigits: int = 12) -> float:
    return float(round(v, ndigits))


def _round_point(p: Any, ndigits: int = 12) -> Any:
    # For R^d points represented as tuple[float, ...], round each coordinate.
    # We do this to prevent that balls where just one dimension is slightly off but produces almost the same ball 
    # will be seen as duplicate by the simplifier
    # This might break mathematical logic, but this is needed for computational feasibility of higher dimensions due to blow up
    # TODO: there are other ways to solve this problem, 
        # e.g. not using open balls for generating, but axis-algined open boxes tuning radii per coordinate or half spaces or a mix of all three
        # this is as open balls can become increasingly rare under some of the common distributions
    if isinstance(p, tuple) and all(isinstance(x, (int, float)) for x in p):
        return tuple(_round_float(float(x), ndigits) for x in p)
    return p

def metric_key(m: object) -> str:
    # include p for LpMetricRd to distinguish L1 vs L2 vs ... vs supnorm
    if isinstance(m, LpMetricRd):
        return f"LpMetricRd(p={m.p})"
    return m.__class__.__qualname__


def event_key(ev: Event[X]) -> tuple:
    """
    Produce a comparable key so we can sort/dedup Union/Intersection parts.

    For AST nodes we generate structural keys.
    For NamedEvent we use its name
    For arbitrary events (e.g. lambdas), we treat them as opaque and key by id().
    That makes dedup stable *within a run*.
    """
    if isinstance(ev, WholeEvent):
        return ("Whole",)
    if isinstance(ev, EmptyEvent):
        return ("Empty",)
    if isinstance(ev, NamedEvent):
        return ("Named", ev.name)
    if isinstance(ev, OpenBall):
        return ("Ball", repr(_round_point(ev.center)), _round_float(ev.radius), metric_key(ev.metric))
    if isinstance(ev, Complement):
        return ("Not", event_key(ev.a))
    if isinstance(ev, Union):
        return ("Or", tuple(event_key(p) for p in ev.parts))
    if isinstance(ev, Intersection):
        return ("And", tuple(event_key(p) for p in ev.parts))
    # opaque leaf
    return ("Opaque", id(ev))


def is_complement_pair(a: Event[X], b: Event[X]) -> bool:
    """
    Detect A vs ¬A pairs.
    Reliable when:
      - b is Complement(a) referencing the same event object, OR
      - keys match structurally: key(b) == ("Not", key(a)) or vice versa.
    """
    ka = event_key(a)
    kb = event_key(b)
    return kb == ("Not", ka) or ka == ("Not", kb)

# ---------- Simplification (only for nodes we can recognize as distinct) ----------

def simplify_event(ev: Event[X]) -> Event[X]:
    # Leaves / opaque events: keep as-is
    # i.e., if it encounters an unknown/opaque event (like a lambda), these are not AST nodes, and thus it treats it as an atomic leaf
    if not isinstance(ev, (WholeEvent, EmptyEvent, NamedEvent, OpenBall, Complement, Union, Intersection)):
        return ev

    # Leaves
    if isinstance(ev, (WholeEvent, EmptyEvent, NamedEvent, OpenBall)):
        return ev

    # we can only do the following rules if we keep in mind that rules like A ∪ ¬A = X only reliably trigger when:
            # ¬A is literally Complement(A) referring to the same A object (or something with the same structural key)
            # for opaque lambdas, there’s no structural equality beyond object identity right now
    # i.e., A ∪ ¬A = X is
        # reliable:
            # A = OpenBall(0.0, 1.0, metric)
            # expr = Union((A, Complement(A))) will evaluate to whole
            # Complement(A).a is A
        # not reliable: 
            # A1 = lambda x: x > 0
            # A2 = lambda x: x > 0
            # logically same set, different function object
            # expr = Union((A1, Complement(A2))) will NOT evaluate to whole as A1 = A2 -> False
    
    # TODO: If we want stronger dedup for arbitrary predicates there are 3 things we can do:
        # wrap these predicates / opaques in a “named” event
            # the event_key will then be able to use the structural key / canonical form of the objects to identify if they are the same
        # only use AST nodes for every sigma algebra we want to build
            # events are then just compared by structural key
        # normal forms / hashing
            # we could hash function bytecode, source text, closure variables, etc.—but it’s hard and could be unsafe
            # we then do not really have mathematical equality if it executes unsafely

        # Complement rules + De Morgan when visible
    if isinstance(ev, Complement):
        a = simplify_event(ev.a)

        if isinstance(a, EmptyEvent):
            return WholeEvent()
        if isinstance(a, WholeEvent):
            return EmptyEvent()
        if isinstance(a, Complement):
            return simplify_event(a.a)

        if isinstance(a, Union):
            return simplify_event(Intersection(tuple(Complement(p) for p in a.parts)))
        if isinstance(a, Intersection):
            return simplify_event(Union(tuple(Complement(p) for p in a.parts)))

        return Complement(a)

    # Union rules
    if isinstance(ev, Union):
        parts = [simplify_event(p) for p in ev.parts]

        # flatten nested unions
        flat: list[Event[X]] = []
        for p in parts:
            if isinstance(p, Union):
                flat.extend(p.parts)
            else:
                flat.append(p)

        # remove ∅; if any Whole => Whole
        filtered: list[Event[X]] = []
        for p in flat:
            if isinstance(p, EmptyEvent):
                continue
            if isinstance(p, WholeEvent):
                return WholeEvent()
            filtered.append(p)

        if not filtered:
            return EmptyEvent()

        # sort + dedup by key (idempotence)
        filtered.sort(key=event_key)
        dedup: list[Event[X]] = []
        seen_keys: set[tuple] = set()
        for p in filtered:
            k = event_key(p)
            if k not in seen_keys:
                dedup.append(p)
                seen_keys.add(k)

        # A ∪ ¬A = Whole
        for i in range(len(dedup)):
            for j in range(i + 1, len(dedup)):
                if is_complement_pair(dedup[i], dedup[j]):
                    return WholeEvent()

        # absorption: A ∪ (A ∩ B) = A
        base_keys = {event_key(p) for p in dedup}
        absorbed: list[Event[X]] = []
        for p in dedup:
            if isinstance(p, Intersection):
                # if intersection contains any base part, drop it
                if any(event_key(q) in base_keys for q in p.parts):
                    continue
            absorbed.append(p)

        if len(absorbed) == 1:
            return absorbed[0]
        absorbed.sort(key=event_key)
        return Union(tuple(absorbed))

    # Intersection rules
    if isinstance(ev, Intersection):
        parts = [simplify_event(p) for p in ev.parts]

        # flatten nested intersections
        flat: list[Event[X]] = []
        for p in parts:
            if isinstance(p, Intersection):
                flat.extend(p.parts)
            else:
                flat.append(p)

        # remove Whole; if any Empty => Empty
        filtered: list[Event[X]] = []
        for p in flat:
            if isinstance(p, WholeEvent):
                continue
            if isinstance(p, EmptyEvent):
                return EmptyEvent()
            filtered.append(p)

        if not filtered:
            return WholeEvent()

        # sort + dedup by key
        filtered.sort(key=event_key)
        dedup: list[Event[X]] = []
        seen_keys: set[tuple] = set()
        for p in filtered:
            k = event_key(p)
            if k not in seen_keys:
                dedup.append(p)
                seen_keys.add(k)

        # A ∩ ¬A = Empty
        for i in range(len(dedup)):
            for j in range(i + 1, len(dedup)):
                if is_complement_pair(dedup[i], dedup[j]):
                    return EmptyEvent()

        # absorption: A ∩ (A ∪ B) = A
        base_keys = {event_key(p) for p in dedup}
        absorbed: list[Event[X]] = []
        for p in dedup:
            if isinstance(p, Union):
                if any(event_key(q) in base_keys for q in p.parts):
                    continue
            absorbed.append(p)

        if len(absorbed) == 1:
            return absorbed[0]
        absorbed.sort(key=event_key)
        return Intersection(tuple(absorbed))

    return ev  # should be unreachable

def generate_event_family(
    generators: list[Event[X]],
    *,
    # max_depth = 0 gives generators and whole and empty when solely AST generators
    # max_depth = 1 gives complements of generators and unions/intersections of pairs of generators 
    # max_depth = 2 gives form depth 1 the complement of a union, union of intersections, intersection involving a complement of a union, etc.
    max_depth: int = 2, 
) -> list[Event[X]]:
    """
    construct a finite closure by generators under Complement/Union/Intersection up to some depth,
    simplifying for computation efficiency and making sure there are no duplicates in the event set along the way.
    this generated event set is a stable test suite for contracts, 
    e.g. catching problems like non-negativity of a measure, or being a probability measure.
    borel sets are the canonical measurable sets for polish spaces (which the state spaces are)
    """
    known: dict[tuple, Event[X]] = {}

    def add(ev: Event[X]) -> None:
        sev = simplify_event(ev)
        known[event_key(sev)] = sev

    add(WholeEvent())
    add(EmptyEvent())
    for g in generators:
        add(g)

    current = [known[event_key(simplify_event(g))] for g in generators]

    for _ in range(max_depth):
        before = set(known.keys())
        new: list[Event[X]] = []

        for A in current:
            new.append(Complement(A))

        for i in range(len(current)):
            for j in range(i, len(current)):
                A, B = current[i], current[j]
                new.append(Union((A, B)))
                new.append(Intersection((A, B)))

        for ev in new:
            add(ev)

        after = set(known.keys())
        frontier_keys = list(after - before)
        current = [known[k] for k in frontier_keys]
        if not current:
            break

    out = list(known.values())
    out.sort(key=event_key)
    return out

