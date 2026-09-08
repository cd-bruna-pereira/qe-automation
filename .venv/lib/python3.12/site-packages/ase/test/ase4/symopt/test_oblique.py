import numpy as np
import pytest

from ase import Atoms
from ase._4.optimize._symopt.relax import (
    AtomsSymmetries,
    SymmetryAdaptedAtoms,
    chol_derivative,
)


def test_oblique():
    """Test against bug where oblique cell vector acquired a z component."""

    cell = np.array(
        [
            6.871958264638166,
            0.0,
            0.0,
            -3.435979132319082,
            6.185040383035336,
            0.0,
            0.0,
            0.0,
            19.74192264375121,
        ]
    ).reshape(3, 3)

    positions = np.array(
        [
            5.569939550493696,
            0.0,
            8.876861983179055,
            1.3020187141444701,
            0.0,
            10.865060660572155,
            1.3062833784489438,
            2.6379468125103815,
            8.350470399495698,
            3.288959089673552,
            0.0,
            7.5,
            0.0,
            4.609899956639001,
            9.870961321875603,
            4.742262510768027,
            3.547093570524955,
            8.350470399495698,
            2.1296957538701418,
            3.547093570524955,
            11.391452244255511,
            3.5829991749646144,
            0.0,
            12.24192264375121,
            3.4359791323190843,
            1.5751404263963347,
            9.870961321875603,
            -1.306283378448943,
            2.6379468125103815,
            11.391452244255511,
        ]
    ).reshape(-1, 3)

    atoms = Atoms(
        numbers=[52, 52, 35, 52, 41, 35, 35, 52, 41, 35],
        cell=cell,
        positions=positions,
        pbc=[True, True, False],
    )

    assert not atoms.cell[2, :2].any()
    assert not atoms.cell[:2, 2].any()

    symmetries = AtomsSymmetries.from_spglib_auto(atoms)
    symmatoms = SymmetryAdaptedAtoms(atoms, symmetries)

    coords = symmatoms.cell_coordinates
    for dM_cc in coords.dM_zcc:
        dC_cv = chol_derivative(coords.M_cc, dM_cc) @ coords.rot_vv.T
        # Bug meant that dC_cv[0, 2] would be nonzero in one case
        assert dC_cv[2, :2] == pytest.approx(0, abs=1e-15)
        assert dC_cv[:2, 2] == pytest.approx(0, abs=1e-15)

    # Let's also verify that actual coordinates do not misbehave:
    x = symmatoms.get_x()
    assert all(x == 0)  # Coordinates start as zero.
    symmatoms.set_x(x + 0.1)

    assert not atoms.cell[2, :2].any()
    assert not atoms.cell[:2, 2].any()
