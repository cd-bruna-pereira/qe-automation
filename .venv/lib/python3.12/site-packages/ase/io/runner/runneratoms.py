#!/usr/bin/env python3
"""Implementation of ASE Atom mimic for temporary fast I/O."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real

import numpy as np

from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import chemical_symbols
from ase.units import Bohr, Hartree

vector_ndim = 2
# Arrays that may be stored in ase.Atoms.arrays but that we treat separately.
SKIPPED_ARRAYS = [
    'numbers',
    'positions',
    'position',
    'element',
    'elements',
    'force',
    'forces',
]

# Translation of property names to clean keys for internal storage.
PROP_NAME_DICT = {
    'positions': 'position',
    'forces': 'force',
    'initial_charges': 'charge',
    'initial_charge': 'charge',
    'charges': 'charge',
    'magnetic_moments': 'magmom',
    'magnetic_moment': 'magmom',
    'initial_magmoms': 'magmom',
    'magmoms': 'magmom',
    'elements': 'element',
    'symbols': 'element',
    'symbol': 'element',
    'hirshfeld_volumes': 'hirshv',
    'hirshfeld_volume': 'hirshv',
}

# Translation of internal keys back to ASE-specific names.
TO_ASE_DICT = {'charge': 'initial_charges', 'magmom': 'initial_magmoms'}


DEFAULT_ATOM_LAYOUT = [('charge', 1), ('atom_energy', 1), ('force', 3)]
# The first four columns in an atom line are always position(3) and element.
ATOM_LAYOUT_OFFSET = 4

CHEMICAL_SYMBOLS_SET = set(chemical_symbols)


class Units(Enum):
    ATOMIC = 'atomic'  # (Ha, Bohr)
    ASE = 'ase'  # (eV, Å)


@dataclass(slots=True)
class RuNNerAtoms:
    """Container for storing data about one atomic structure.

    This class is intended for fast I/O of structural data. It is required as
    the builtin ASE `Atoms` and `Atom` classes are a bottleneck for large data
    files. This is mostly because adding an `Atom` to an `Atoms` object over
    and over again comes with a lot of overhead because the attached arrays
    have to be checked and copied.
    In constrast, it is very efficient to store atomic positions, symbols, etc.
    in long lists and simply create one ASE `Atoms` object once all atoms have
    been collected. In summary, `RuNNerAtoms` is simply a container to hold all
    these lists in one convenient place.
    """

    # Keep track of the units used to fill the internal storage.
    input_units: Units = Units.ASE
    # Store the length of each property in `self.atom_arrays`.
    atom_layout: list[tuple[str, int]] = field(default_factory=list)
    # Store the total number of columns specified by `atom_layout`.
    num_atom_columns: int = 0

    # Mandatory arrays.
    positions: list[list[float]] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    cell: list[list[float]] = field(default_factory=list)

    # Properties that end up in the attached SinglePointCalculator.
    energy: float | None = None
    forces: list[list[float]] | None = None

    # Properties that end up in atoms.arrays
    atom_arrays: dict[str, list[float | list[float]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # Properties that end up in atoms.info
    total_charge: float = 0.0

    def to_ase_atoms(self) -> Atoms:
        """Convert the object to an ASE Atoms object."""

        atoms = Atoms(
            positions=self.positions,
            symbols=self.symbols,
        )

        if self.cell:
            if len(self.cell) != 3:
                raise ValueError(
                    'RuNNer requires the specification of 3 lattice vectors.'
                )

            atoms.set_cell(self.cell)
            atoms.pbc = True

        for name, arr in self.atom_arrays.items():
            if name in SKIPPED_ARRAYS:
                continue

            name = TO_ASE_DICT.get(name, name)
            atoms.set_array(name, arr, dtype=np.float64)

        atoms.info['total_charge'] = self.total_charge

        atoms.calc = SinglePointCalculator(
            atoms,
            energy=self.energy,
            forces=self.forces,
        )

        return atoms

    def _validate_atom_arrays(self) -> None:
        for name, arr in self.atom_arrays.items():
            if len(arr) != len(self.positions):
                raise ValueError(
                    f'The count ({len(arr)}) of custom property "{name}"',
                    ' is not equal to the number of atoms',
                    f' ({len(self.positions)}) in this structure',
                )
        self.num_atom_columns = (
            sum([i for _, i in self.atom_layout]) + ATOM_LAYOUT_OFFSET
        )

    # This type-hint is not quite correct for inherited classes:
    # from Python 3.11 use -> typing.Self
    @classmethod
    def from_ase_atoms(
        cls,
        atoms: Atoms,
        input_units: Units = Units.ASE,
    ) -> RuNNerAtoms:
        """Convert an ASE Atoms object to this class."""
        # Construct the `atom_layout` for mapping arbitrary arrays.
        atom_layout = []
        for name, arr in atoms.arrays.items():
            if name in SKIPPED_ARRAYS:
                continue

            name = PROP_NAME_DICT.get(name, name)

            if arr.ndim == 1:
                length = 1
            elif arr.ndim == vector_ndim:
                length = arr.shape[1]
            else:
                raise ValueError(
                    f'{arr.ndim - 1}-dimensional custom properties not',
                    'supported by the input.data format.',
                )

            atom_layout.append((name, length))

        temp_atoms = cls(
            positions=atoms.positions.tolist(),
            symbols=list(atoms.symbols),
            input_units=input_units,
            atom_layout=atom_layout,
        )

        if any(atoms.pbc):
            temp_atoms.cell = atoms.cell

        for name, arr in atoms.arrays.items():
            if name not in SKIPPED_ARRAYS and issubclass(arr.dtype.type, Real):
                name = PROP_NAME_DICT.get(name, name)
                temp_atoms.atom_arrays[name] = arr.tolist()

        temp_atoms._validate_atom_arrays()

        if atoms.calc is not None:
            if 'energy' in atoms.calc.results:
                temp_atoms.energy = atoms.calc.results['energy']
            if 'forces' in atoms.calc.results:
                temp_atoms.forces = atoms.calc.results['forces'].tolist()

        if 'total_charge' in atoms.info:
            temp_atoms.total_charge = atoms.info['total_charge']

        return temp_atoms

    def convert(self, output_units: Units) -> None:
        """Convert values in-place from `self.input_units` to `output_units`."""
        if self.input_units is output_units:
            return

        if self.input_units is Units.ATOMIC and output_units is Units.ASE:
            length_factor = Bohr
            energy_factor = Hartree
            force_factor = Hartree / Bohr
        elif self.input_units is Units.ASE and output_units is Units.ATOMIC:
            length_factor = 1.0 / Bohr
            energy_factor = 1.0 / Hartree
            force_factor = Bohr / Hartree
        else:
            raise ValueError(
                f'Unsupported conversion: {self.input_units} -> {output_units}'
            )

        if self.positions is not None:
            self.positions = [
                [i * length_factor for i in xyz] for xyz in self.positions
            ]

        if self.cell is not None:
            self.cell = [[i * length_factor for i in xyz] for xyz in self.cell]

        if self.energy is not None:
            self.energy *= energy_factor

        if self.forces is not None:
            self.forces = [
                [i * force_factor for i in xyz] for xyz in self.forces
            ]

        self.input_units = output_units

    def add_atom(self, values: list[str]):
        """Add information from a single atom line to storage."""
        offset = ATOM_LAYOUT_OFFSET  # start in the field after the element.

        # Input validation.
        if values[3] not in CHEMICAL_SYMBOLS_SET:
            raise ValueError(f"Unknown element '{values[3]}'.")

        if len(values) != self.num_atom_columns:
            raise ValueError(
                f'Wrong format of atom line. Expected {self.num_atom_columns} '
                f'columns after the atom statement but got {len(values)}.'
            )

        self.positions.append([float(v) for v in values[0:3]])
        self.symbols.append(values[3])

        for name, length in self.atom_layout:
            if length == 1:
                prop_vals: float | list[float] = float(values[offset])
            else:
                prop_vals = [float(v) for v in values[offset : offset + length]]

            match name:
                case 'force':
                    # Forces are treated separately since they are stored on the
                    # calculator and not in the ase.Atoms.arrays dictionary.
                    if self.forces is None:
                        self.forces = []
                    self.forces.append(
                        [float(v) for v in values[offset : offset + length]]
                    )
                case _:
                    self.atom_arrays[name].append(prop_vals)

            offset += length

    def add_lattice_vector(self, values: list[str]):
        """Add a single lattice vector to storage."""
        if len(values) != 3:
            raise ValueError(
                'Lattice vectors must be exactly 3 components long.'
            )

        if len(self.cell) == 3:
            raise ValueError('Cannot store more than three lattice vectors.')

        self.cell.append([float(v) for v in values])

    def parse_line(self, line: str) -> None:
        """Parse a single line from input.data."""
        line = line.strip()
        if not line or line.startswith('#'):
            return

        fields = line.split()
        keyword, values = fields[0], fields[1:]

        match keyword:
            case 'atom':
                self.add_atom(values)

            case 'lattice':
                self.add_lattice_vector(values)

            case 'energy':
                self.energy = float(values[0])

            case 'charge':
                self.total_charge = float(values[0])
