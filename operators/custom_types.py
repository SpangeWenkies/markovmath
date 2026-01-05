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
    Annotated,
    TYPE_CHECKING,
)

X = TypeVar("X")
E = TypeVar("E")
DensityLike = TypeVar("DensityLike")
LawLike = TypeVar("LawLike")
Observable = Callable[[X], float]
Scalar = float | complex
DensityVector: TypeAlias = Sequence[float]
KeyFn = Callable[[X], Hashable]

# use np.array as these are not hashable, do Point = tuple[float, ...] of length d
# note we need python version >= 3.9 for this, otherwise we need Tuple from typing
# maybe later on we need a tensor instead of a matrix for certain applications

Vector: TypeAlias = tuple[float, ...]
MutableVector: TypeAlias = list[float]
VectorLike: TypeAlias = Vector | MutableVector

Matrix: TypeAlias = tuple[tuple[float, ...], ...]
MutableMatrix: TypeAlias = list[list[float]]
MatrixLike: TypeAlias = Matrix | MutableMatrix

EigenDecomposition: TypeAlias = tuple[MutableVector, MutableMatrix]

PointRd: TypeAlias = tuple[float, ...]

Density: TypeAlias = Callable[[X], float]

NonNegativeFloat = Annotated[float, ">=0"]
PositiveFloat = Annotated[float, ">0"]
PositiveRd = Annotated[PointRd, ">0"]
NonNegativeRd = Annotated[PointRd, ">=0"]

if TYPE_CHECKING:
    from .generators import (
        ClosedFormGenerator,
        FiniteStateCTMCGenerator,
        SampledGenerator,
    )

Generator: TypeAlias = (
    "SampledGenerator[X] | ClosedFormGenerator[X] | FiniteStateCTMCGenerator[X]"
)


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
