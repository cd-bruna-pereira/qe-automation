from io import StringIO
from shutil import copyfile

import numpy as np
import pytest

# from ase import Atoms
from ase.io import read  # , iread
from ase.io.vasp import read_vasp

BUF_WITH_HASHES = """\
Na1
   1.0000000000000000
     5.9524044248000987   -0.0007582713987269   -0.0012396575089365
    -0.0014963523464428    6.3481246432503076   -0.0209332994306307
    -0.0014359618019133   -3.1516690207714437    5.5105006246735595
  Na_pv/6a2f546d
               1
Direct
  0.9711350580843501  0.0342911094149940  0.0033526422550807
"""


@pytest.fixture()
def outcar(datadir):
    return datadir / 'vasp' / 'OUTCAR_example_1'


@pytest.fixture()
def poscar_no_species(datadir):
    return datadir / 'vasp' / 'POSCAR_example_1'


def test_read_poscar_no_species(outcar, poscar_no_species, tmp_path):
    copyfile(outcar, tmp_path / 'OUTCAR')
    copyfile(poscar_no_species, tmp_path / 'POSCAR')

    at_outcar = read(outcar)
    at_poscar = read(tmp_path / 'POSCAR')

    assert len(at_outcar) == len(at_poscar)
    assert np.all(np.isclose(at_outcar.cell, at_poscar.cell))
    assert np.all(np.isclose(at_outcar.positions, at_poscar.positions))
    assert np.all(at_outcar.numbers == at_poscar.numbers)


def test_read_atom_types_with_hashes() -> None:
    """Test if `read_vasp` can read atom types with POTCAR hashes."""
    atoms = read_vasp(StringIO(BUF_WITH_HASHES))
    numbers_ref = [11]
    np.testing.assert_allclose(atoms.numbers, numbers_ref)
