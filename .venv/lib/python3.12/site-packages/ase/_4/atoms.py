"""ASEv4 `Atoms` class."""

from ase._4.calculators.results import CalculationResults
from ase.atoms import Atoms as V3Atoms
from ase.atoms import _LimitedAtoms
from ase.outputs import ArrayProperty, all_outputs


class Atoms(_LimitedAtoms):
    """ASEv4 Atoms class."""

    @classmethod
    def from_v3atoms(cls, v3atoms: V3Atoms):
        """Make ASEv4 Atoms from ASEv3 Atoms."""
        return cls(
            symbols=v3atoms.symbols,
            positions=v3atoms.positions,
            cell=v3atoms.cell,
            pbc=v3atoms.pbc,
        )

    def store(
        self,
        results: CalculationResults,
        prefix: str = '',
        suffix: str = '',
    ) -> None:
        """Stores `results` in `Atoms.info` and and `Atoms.arrays`.

        Parameters
        ----------
        results : `CalculationResults`
            Properties to be saved.
        prefix : str
            Prefix to the names of the properties.
        suffix : str
            Suffix to the names of the properties.

        """
        for prop_name, prop_val in results.properties.items():
            output_type = all_outputs[prop_name]
            if (
                isinstance(output_type, ArrayProperty)
                and output_type.shapespec[0] == 'natoms'
            ):
                self.arrays[prefix + prop_name + suffix] = prop_val
            else:
                self.info[prefix + prop_name + suffix] = prop_val

    def get_potential_energy(self, **kwargs):
        raise NotImplementedError(
            "Use calculator.evaluate(atoms, properties=['energy']) in ASE v4."
        )

    def get_potential_energies(self, **kwargs):
        raise NotImplementedError(
            "Use calculator.evaluate(atoms, properties=['energies']) in ASE v4."
        )

    def get_forces(self, **kwargs):
        raise NotImplementedError(
            "Use calculator.evaluate(atoms, properties=['forces']) in ASE v4."
        )

    def get_stress(self, **kwargs):
        raise NotImplementedError(
            "Use calculator.evaluate(atoms, properties=['stress']) in ASE v4."
        )

    def get_stresses(self, **kwargs):
        raise NotImplementedError(
            "Use calculator.evaluate(atoms, properties=['stresses']) in ASE v4."
        )

    def get_charges(self, **kwargs):
        raise NotImplementedError(
            "Use calculator.evaluate(atoms, properties=['charges']) in ASE v4."
        )

    def get_magnetic_moments(self, **kwargs):
        raise NotImplementedError(
            "Use calculator.evaluate(atoms, properties=['magmoms']) in ASE v4."
        )

    def get_magnetic_moment(self, **kwargs):
        raise NotImplementedError(
            "Use calculator.evaluate(atoms, properties=['magmom']) in ASE v4."
        )

    def get_dipole_moment(self, **kwargs):
        raise NotImplementedError(
            "Use calculator.evaluate(atoms, properties=['dipole']) in ASE v4."
        )

    def get_total_energy(self, **kwargs):
        raise NotImplementedError(
            "Use calculator.evaluate(atoms, properties=['energy']) in ASE v4 "
            'and add atoms.get_kinetic_energy().'
        )

    def get_properties(self, *args, **kwargs):
        raise NotImplementedError(
            'Use calculator.evaluate(atoms, properties=[...]) in ASE v4.'
        )
