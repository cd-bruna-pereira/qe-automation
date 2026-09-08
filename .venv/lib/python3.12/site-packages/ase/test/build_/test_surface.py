"""Tests for `surface`"""

import math
from contextlib import nullcontext

import numpy as np
import pytest

from ase import Atom, Atoms
from ase.build import (
    add_adsorbate,
    bulk,
    fcc111,
    fcc211,
    graphene,
    mx2,
    surface,
)


def test_surface():
    """Test general"""
    atoms = fcc211('Au', (3, 5, 8), vacuum=10.0)
    assert len(atoms) == 120

    atoms = atoms.repeat((2, 1, 1))
    assert np.allclose(atoms.get_distance(0, 130), 2.88499566724)

    atoms = fcc111('Ni', (2, 2, 4), orthogonal=True)
    add_adsorbate(atoms, 'H', 1, 'bridge')
    add_adsorbate(atoms, Atom('O'), 1, 'fcc')
    add_adsorbate(atoms, Atoms('F'), 1, 'hcp')

    # The next test ensures that a simple string of multiple atoms
    # cannot be used, which should fail with a KeyError that reports
    # the name of the molecule due to the string failing to work with
    # Atom().
    failed = False
    try:
        add_adsorbate(atoms, 'CN', 1, 'ontop')
    except KeyError as err:
        failed = True
        assert err.args[0] == 'CN'
    assert failed

    # This test ensures that the default periodic behavior remains unchanged
    cubic_fcc = bulk('Al', a=4.05, cubic=True)
    surface_fcc = surface(cubic_fcc, (1, 1, 1), 3)

    assert list(surface_fcc.pbc) == [True, True, False]
    assert surface_fcc.cell[2][2] == 0

    # This test checks the new periodic option
    cubic_fcc = bulk('Al', a=4.05, cubic=True)
    surface_fcc = surface(cubic_fcc, (1, 1, 1), 3, periodic=True)

    assert list(surface_fcc.pbc) == [True, True, True]
    expected_length = 4.05 * 3**0.5  # for FCC with a=4.05
    assert math.isclose(surface_fcc.cell[2][2], expected_length)

    # This test checks the tags
    print(surface_fcc.get_tags())
    tags = np.array([3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1])
    np.testing.assert_array_equal(surface_fcc.get_tags(), tags)


@pytest.mark.parametrize('vacuum', [None, 10.0])
def test_others(vacuum):
    """Test if other types of `surface` functions (at least) run."""
    mx2(kind='2H', vacuum=vacuum)
    mx2(kind='1T', vacuum=vacuum)
    graphene(vacuum=vacuum)
    graphene(thickness=0.5, vacuum=vacuum)


@pytest.mark.parametrize(
    'vacuum, a',
    [(None, 3.77), (0.0, 3.77), (-1.0, 3.77), (10.0, 3.74)],
)
def test_surface_layer_splitting(vacuum, a):
    """
    Test that building an fcc(111) surface from a multielement lattice
    does not erroneously split atoms into an extra layer due to
    floating-point precision issues in boundary wrapping.
    """
    n_layers = 3

    # Create Cu3Au bulk cell
    cu3au = Atoms(
        'Cu3Au',
        scaled_positions=[
            (0.5, 0.5, 0),
            (0.5, 0, 0.5),
            (0, 0.5, 0.5),
            (0, 0, 0),
        ],
        cell=[a, a, a],
        pbc=True,
    )

    # Generate surface with enough vacuum so boundaries don't fold atoms
    if vacuum is not None and vacuum <= 0.0:
        # Expect a FutureWarning for any vacuum <= 0.0
        expectation = pytest.warns(
            FutureWarning, match='Passing vacuum <= 0 is deprecated'
        )
    else:
        expectation = nullcontext()

    with expectation:
        slab = surface(
            cu3au,
            indices=(1, 1, 1),
            layers=n_layers,
            vacuum=vacuum,
            periodic=True,
        )

    slab.wrap()

    # Extract Z coordinates
    z_coords = slab.positions[:, 2]

    # Group atoms by height to count the actual physical layers.
    # Rounding to 3 decimals absorbs legitimate minor numeric noise,
    # but easily catches distinct layers (which are ~2 Angstroms apart).
    unique_z = np.unique(np.round(z_coords, decimals=3))

    if vacuum == 0.0:
        # If vacuum == 0.0, we expect potential floating point boundary
        # splitting due to the uncentered fallback behavior.
        assert len(unique_z) != n_layers
    else:
        # Assert we have exactly the requested number of layers
        assert len(unique_z) == n_layers, (
            f'Floating point boundary error: Requested {n_layers} layers, '
            f'but generated a slab with {len(unique_z)} distinct Z-heights '
            f'(Z-levels: {unique_z}).'
        )
