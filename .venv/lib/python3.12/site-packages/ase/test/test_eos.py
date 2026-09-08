"""Tests for the EOS fitting."""

import numpy as np
import pytest

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.eos import EquationOfState as EOS
from ase.eos import eos_names


def test_eos() -> None:
    """Test EOS fitting."""
    atoms = bulk('Al', 'fcc', a=4.0, orthorhombic=True)
    atoms.calc = EMT()
    cell = atoms.get_cell()

    volumes = []
    energies = []
    for x in np.linspace(0.98, 1.01, 5):
        atoms.set_cell(cell * x, scale_atoms=True)
        volumes.append(atoms.get_volume())
        energies.append(atoms.get_potential_energy())

    results = []
    for name in eos_names:
        eos = EOS(volumes, energies, name)
        v, e, b = eos.fit()
        print(f'{name:20} {v:.8f} {e:.8f} {b:.8f} ')
        assert v == pytest.approx(3.18658700e01, abs=4e-4)
        assert e == pytest.approx(-9.76187802e-03, abs=1e-6)
        assert b == pytest.approx(2.46812688e-01, abs=2e-4)
        results.append((v, e, b))

    print(np.ptp(results, 0))
    print(np.mean(results, 0))
