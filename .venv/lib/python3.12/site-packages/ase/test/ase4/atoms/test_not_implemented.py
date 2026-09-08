import pytest


def test_get_potential_energy_raises(v4atoms):
    with pytest.raises(NotImplementedError, match='calculator.evaluate'):
        v4atoms.get_potential_energy()


def test_get_forces_raises(v4atoms):
    with pytest.raises(NotImplementedError, match='calculator.evaluate'):
        v4atoms.get_forces()


def test_get_stress_raises(v4atoms):
    with pytest.raises(NotImplementedError, match='calculator.evaluate'):
        v4atoms.get_stress()


def test_get_potential_energies_raises(v4atoms):
    with pytest.raises(NotImplementedError, match='calculator.evaluate'):
        v4atoms.get_potential_energies()


def test_get_stresses_raises(v4atoms):
    with pytest.raises(NotImplementedError, match='calculator.evaluate'):
        v4atoms.get_stresses()


def test_get_charges_raises(v4atoms):
    with pytest.raises(NotImplementedError, match='calculator.evaluate'):
        v4atoms.get_charges()


def test_get_magnetic_moments_raises(v4atoms):
    with pytest.raises(NotImplementedError, match='calculator.evaluate'):
        v4atoms.get_magnetic_moments()


def test_get_magnetic_moment_raises(v4atoms):
    with pytest.raises(NotImplementedError, match='calculator.evaluate'):
        v4atoms.get_magnetic_moment()


def test_get_dipole_moment_raises(v4atoms):
    with pytest.raises(NotImplementedError, match='calculator.evaluate'):
        v4atoms.get_dipole_moment()


def test_get_total_energy_raises(v4atoms):
    with pytest.raises(NotImplementedError, match='calculator.evaluate'):
        v4atoms.get_total_energy()


def test_get_properties_raises(v4atoms):
    with pytest.raises(NotImplementedError, match='calculator.evaluate'):
        v4atoms.get_properties(['energy'])
