import numpy as np
import pytest

from ase import Atoms
from ase.build import bulk, fcc111
from ase.calculators.fd import (
    calculate_numerical_forces,
    calculate_numerical_stress,
)

potentials = (
    'Pt_u3.eam',
    'NiAlH_jea.eam.alloy',
    'NiAlH_jea.eam.fs',
    'AlCu.adp',
)


@pytest.mark.calculator('eam')
@pytest.mark.calculator_lite()
def test_eam_run(factory):
    with open(f'{factory.factory.potentials_path}/Pt_u3.eam') as fd:
        eam = factory.calc(potential=fd, form='eam', elements=['Pt'])
    slab = fcc111('Pt', size=(4, 4, 2), vacuum=10.0)
    slab.calc = eam

    assert abs(-164.277599313 - slab.get_potential_energy()) < 1e-8
    assert abs(6.36379627645 - np.linalg.norm(slab.get_forces())) < 1e-8


@pytest.mark.parametrize('with_elements', (False, True))
@pytest.mark.parametrize('potential', potentials)
@pytest.mark.calculator('eam')
@pytest.mark.calculator_lite()
def test_read_potential(factory, potential: str, with_elements: bool):
    """Test if the potential can be read without errors."""
    element = potential[:2]
    potential = f'{factory.factory.potentials_path}/{potential}'
    if with_elements:
        calc = factory.calc(potential=potential, elements=[element])
    else:
        calc = factory.calc(potential=potential)
    atoms = bulk(element)
    atoms.calc = calc
    atoms.get_potential_energy()

    # test forces against numerical forces
    forces = atoms.get_forces()
    numerical_forces = calculate_numerical_forces(atoms, eps=1e-5)
    np.testing.assert_allclose(forces, numerical_forces, atol=1e-5)

    # test stress against numerical stress
    stress = atoms.get_stress()
    numerical_stress = calculate_numerical_stress(atoms, eps=1e-5)
    np.testing.assert_allclose(stress, numerical_stress, atol=1e-5)


def _make_atoms(potential: str) -> Atoms:
    if potential == 'Pt_u3.eam':
        return bulk('Pt')
    if potential in {'NiAlH_jea.eam.alloy', 'NiAlH_jea.eam.fs'}:
        atoms = bulk('NiAl', 'cesiumchloride', a=3.0)
        atoms += Atoms('H', positions=[[0.0, 0.0, 1.5]])
        return atoms
    if potential == 'AlCu.adp':
        atoms = bulk('Cu', cubic=True)
        atoms.symbols[0] = 'Al'
        return atoms
    raise ValueError(potential)


@pytest.mark.parametrize('potential', potentials)
@pytest.mark.calculator('eam')
@pytest.mark.calculator_lite()
def test_write_potential(tmp_path, factory, potential: str) -> None:
    """Test if `write_potential` reproduces the same energy."""
    atoms = _make_atoms(potential)

    potential_ref = f'{factory.factory.potentials_path}/{potential}'
    atoms.calc = factory.calc(potential=potential_ref)
    energy_ref = atoms.get_potential_energy()

    potential_tmp = f'{tmp_path}/{potential}'
    atoms.calc.write_potential(potential_tmp)
    atoms.calc = factory.calc(potential=potential_tmp)
    energy_tmp = atoms.get_potential_energy()

    assert energy_tmp == pytest.approx(energy_ref)
