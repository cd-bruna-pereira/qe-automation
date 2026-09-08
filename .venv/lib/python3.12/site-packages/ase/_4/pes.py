import numpy as np

from ase._4.atoms import Atoms
from ase._4.calculators.calculator import BaseCalculator
from ase._4.calculators.results import CalculationResults
from ase.calculators.calculator import PropertyNotImplementedError


class PotentialEnergySurface:
    """Caches and manages evaluations of atoms.

    Used by geometry optimisation, molecular dynamics, and
    similar, wherever caching is required.

    Replacing ``pes.atoms`` or ``pes.calc`` invalidates the cache
    automatically. Mutating the existing atoms in place, however,
    bypasses the cache, so such operations must go through
    PotentialEnergySurface methods (e.g., ``pes.set_positions``)
    and not directly (NOT ``atoms.set_positions``).

    DEV: Some requested properties are pre-set at initialization,
         but we might want to rethink this.

    """

    def __init__(
        self,
        atoms: Atoms,
        calc: BaseCalculator,
        requested_properties: tuple[str, ...] | None = None,
    ):
        """Initialize the potential energy surface.

        Parameters
        ----------
        atoms: Atoms
            Atomic configuration to evaluate.
        calc: BaseCalculator
            Calculator used to compute properties of ``atoms``.
        requested_properties: tuple of str, optional
            Properties to request from ``calc`` on each evaluate().
            Defaults to ``('free_energy', 'forces')`` if the calculator
            implements ``free_energy``, otherwise ``('energy', 'forces')``.
        """
        self.atoms = atoms
        self.calc = calc

        # requested properties follow general ASEv3 behaviour
        # for picking "energy" vs "free_energy". Likely to be
        # reconsidered in the future, but keeping it faithfully here.
        if requested_properties is None:
            if 'free_energy' in calc.implemented_properties:
                requested_properties = ('free_energy', 'forces')
            else:
                requested_properties = ('energy', 'forces')
        self.requested_properties = tuple(requested_properties)
        unsupported = set(self.requested_properties) - set(
            calc.implemented_properties
        )
        if unsupported:
            raise ValueError(
                f'requested_properties {sorted(unsupported)} not in '
                f'calc.implemented_properties '
                f'{sorted(calc.implemented_properties)}'
            )
        self.results: CalculationResults | None = None

    @property
    def atoms(self) -> Atoms:
        return self._atoms

    @atoms.setter
    def atoms(self, atoms: Atoms) -> None:
        self.invalidate()
        self._atoms = atoms

    @property
    def calc(self) -> BaseCalculator:
        return self._calc

    @calc.setter
    def calc(self, calc: BaseCalculator) -> None:
        self.invalidate()
        self._calc = calc

    def evaluate(self) -> CalculationResults:
        if self.results is None:
            self.results = self.calc.evaluate(
                self.atoms, properties=list(self.requested_properties)
            )
        return self.results

    def invalidate(self) -> None:
        self.results = None

    def get_positions(self) -> np.ndarray:
        return self.atoms.get_positions()

    def set_positions(self, x: np.ndarray) -> None:
        """Assumes x is positions-shaped. Invalidates the cache."""
        self.invalidate()
        self.atoms.set_positions(x)

    def get_property(self, prop: str):
        results = self.evaluate()
        if prop not in results.properties:
            raise PropertyNotImplementedError(
                f'Tried to get {prop} which has not been evaluated. '
                f'Available properties are {list(results.properties.keys())}.'
            )
        return results.properties[prop]
