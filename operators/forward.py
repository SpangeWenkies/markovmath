from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Optional, TypeVar

from core_interfaces import Density, DensityEvolution, LawEvolution, ProbabilityMeasure

X = TypeVar("X")
E = TypeVar("E")


@dataclass(slots=True)
class ForwardEquation(Generic[X, E]):
    """
    Forward Kolmogorov/Fokker-Planck wrapper that dispatches on representation.

    Provide either a law solver or a density solver (or both).
    """

    law_solver: Optional[LawEvolution[X, E]] = None
    density_solver: Optional[DensityEvolution[X]] = None

    def forward_law_step(
        self, mu0: ProbabilityMeasure[X, E], t: float
    ) -> ProbabilityMeasure[X, E]:
        if self.law_solver is None:
            raise ValueError("No law solver configured for ForwardEquation.")
        return self.law_solver.evolve_law(mu0, t)

    def forward_density_step(self, p0: Density[X], t: float) -> Density[X]:
        if self.density_solver is None:
            raise ValueError("No density solver configured for ForwardEquation.")
        return self.density_solver.evolve_density(p0, t)

    def forward(
        self, initial: ProbabilityMeasure[X, E] | Density[X], t: float
    ) -> ProbabilityMeasure[X, E] | Density[X]:
        if self._is_law(initial):
            return self.forward_law_step(initial, t)
        return self.forward_density_step(initial, t)

    @staticmethod
    def _is_law(obj: Any) -> bool:
        return hasattr(obj, "sample") or hasattr(obj, "measure")