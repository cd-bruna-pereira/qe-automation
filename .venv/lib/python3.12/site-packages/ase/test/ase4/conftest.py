"""`conftest.py` for ASEv4."""

import numpy as np
import pytest

from ase import Atoms as V3Atoms
from ase._4.atoms import Atoms as V4Atoms
from ase._4.calculators.calculator import BaseCalculator
from ase._4.calculators.emt import EMT
from ase.build import bulk
from ase.outputs import Properties

# ---------------------------
# CalculationResults fixtures
# ---------------------------


@pytest.fixture
def metadata() -> dict:
    """ "Input to create CalculationResults from"""
    return {'calculator_name': 'test_calculator', 'calculator_version': 'v3.2'}


@pytest.fixture(name='properties_dict')
def fixture_properties_dict() -> dict:
    """ "Input to create CalculationResults from"""
    data = {
        'energy': 3.14,
        'forces': np.arange(15).reshape((5, 3)),
        'stress': np.arange(6),
    }
    return data


@pytest.fixture(params=['dict', 'Properties'])
def properties(
    request: pytest.FixtureRequest,
    properties_dict: dict,
) -> dict | Properties:
    """Two ways of initialising properties in CalculationResults."""
    if request.param == 'dict':
        return properties_dict
    if request.param == 'Properties':
        return Properties(properties_dict)
    raise RuntimeError


# ---------------------
# v4Calculator fixtures
# ---------------------


@pytest.fixture
def atoms() -> V3Atoms:
    atoms = bulk('Cu', 'fcc', a=3.6) * (2, 2, 2)
    atoms.rattle(stdev=0.01, seed=42)
    return atoms


# a bit hacky for now
@pytest.fixture
def v4atoms(atoms) -> V4Atoms:
    return V4Atoms.from_v3atoms(atoms)


@pytest.fixture
def calculator() -> BaseCalculator:
    calculator = EMT()
    return calculator
