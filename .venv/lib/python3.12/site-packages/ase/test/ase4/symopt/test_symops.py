import pytest

from ase.build import bulk

pytestmark = pytest.mark.filterwarnings('ignore:Set OLD_ERROR_HANDLING')


@pytest.fixture
def diamond():
    return bulk('C')


def spglib_dataset(atoms):
    import spglib

    return spglib.get_symmetry_dataset(
        (atoms.cell, atoms.get_scaled_positions(), atoms.numbers)
    )


def symop_dict(rotations, translations):
    assert len(rotations) == len(translations)
    dct = {}
    for rot, trans in zip(rotations, translations):
        key = tuple(rot.reshape(9).tolist())
        assert key not in dct
        # Here we should perhaps normalize 1.0 to 0.0 (unless non-PBC)
        # and truncate things like 1e-16.
        dct[key] = trans
    return dct


@pytest.mark.skip(
    'Will pass only gpaw dev version due to translations sign change'
)
def test_compare_symops_gpaw_vs_spglib(diamond):
    """Ensure that we understand the differences between spglib/GPAW symops."""
    pytest.importorskip('gpaw')
    from gpaw.new.symmetry import create_symmetries_object

    symmetry = create_symmetries_object(diamond, symmorphic=False)
    dataset = spglib_dataset(diamond)

    gpaw_symops = symop_dict(symmetry.rotation_scc, symmetry.translation_sc)
    spglib_symops = symop_dict(
        dataset.rotations.transpose(0, 2, 1), dataset.translations
    )

    assert set(gpaw_symops) == set(spglib_symops)
    for key in gpaw_symops:
        assert spglib_symops[key] == pytest.approx(gpaw_symops[key])
