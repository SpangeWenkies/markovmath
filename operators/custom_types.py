from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import (
    Callable,
    Hashable,
    Iterator,
    Protocol,
    Sequence,
    TypeAlias,
    TypeVar,
    Generic,
    runtime_checkable,
)

X = TypeVar("X")

Observable = Callable[[X], float]

Scalar = float | complex

DensityVector: TypeAlias = Sequence[float]

KeyFn = Callable[[X], Hashable]


class GeneratorSource(str, Enum):
    CLOSED_FORM = "closed_form"
    SAMPLED = "sampled"


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