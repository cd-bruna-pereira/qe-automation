import numpy as np
import pytest

from ase import Atoms
from ase._4.optimize.bfgs import BFGSMethod
from ase._4.optimize.frechet import FrechetOptimizable
from ase._4.optimize.lbfgs import LBFGSMethod
from ase._4.optimize.method import OptimizerMethod
from ase._4.optimize.optimizable import Optimizable4, OptimizableAtoms
from ase._4.optimize.run import (
    Optimizer,
    read_images,
    write_restartfile,
)
from ase.calculators.emt import EMT
from ase.filters import FrechetCellFilter
from ase.optimize.bfgs import BFGS as OldBFGS
from ase.optimize.cellawarebfgs import CellAwareBFGS
from ase.optimize.lbfgs import LBFGS as OldLBFGS
from ase.optimize.optimize import Optimizer as OldOptimizer


def setup_surface():
    from ase.build import fcc111

    rng = np.random.RandomState(42)
    atoms = fcc111('Au', size=(1, 2, 2), vacuum=5.0)
    atoms.rattle(stdev=0.01, rng=rng)
    cell = atoms.get_cell()
    cell[:2, :2] += 0.05 * rng.random((2, 2))
    atoms.set_cell(cell, scale_atoms=True)
    atoms.calc = EMT()
    return atoms


algorithm_names = [
    'BFGS',
    'LBFGS',
]


def _make_optimizer_ase3(
    atoms: Atoms,
    algorithm_name: str,
) -> OldOptimizer:
    alpha = 70.0
    if algorithm_name == 'BFGS':
        return OldBFGS(atoms, alpha=alpha)
    if algorithm_name == 'LBFGS':
        return OldLBFGS(atoms, alpha=alpha)
    raise ValueError(algorithm_name)


def _make_algorithm_ase4(
    target: Optimizable4,
    algorithm_name: str,
) -> OptimizerMethod:
    alpha = 70.0
    if algorithm_name == 'BFGS':
        return BFGSMethod(target.initial_hessian(alpha=alpha))
    if algorithm_name == 'LBFGS':
        initial_inverse_hessian = target.initial_inverse_hessian(alpha=alpha)
        return LBFGSMethod(initial_inverse_hessian=initial_inverse_hessian)
    raise ValueError(algorithm_name)


@pytest.mark.parametrize('algorithm_name', algorithm_names)
def test_optimizer(algorithm_name: str) -> None:
    """Test if the ASE3- and ASE4-style optimizers give the same energy.

    At present, BFGS and LBGFS are tested.
    Within the history size, LBFGS should provide mathematically the same
    results as BFGS, and therefore we can use the same reference energy.
    """
    fmax = 0.01

    atoms_opt3 = setup_surface()
    opt3 = _make_optimizer_ase3(atoms_opt3, algorithm_name)
    for opt3_i, _ in enumerate(opt3.irun(fmax=fmax)):
        pass

    atoms_opt4 = setup_surface()
    target = OptimizableAtoms(atoms_opt4, fmax=fmax)
    method = _make_algorithm_ase4(target, algorithm_name)
    opt4 = Optimizer(target, method)
    step = opt4.run()

    assert step.gradient_obj.converged

    assert opt3_i == 10
    assert step.i == 10

    ref_energy = 0.9133826276
    assert atoms_opt3.get_potential_energy() == pytest.approx(ref_energy)
    assert atoms_opt4.get_potential_energy() == pytest.approx(ref_energy)


def old_frechet_energy(atoms, fmax, smax):
    atoms = setup_surface()
    bfgs = CellAwareBFGS(
        FrechetCellFilter(atoms, exp_cell_factor=1.0, mask=[1, 1, 0, 0, 0, 1])
    )
    bfgs.run(fmax=0.001, smax=0.0001)
    return atoms.get_potential_energy()


@pytest.mark.parametrize('algorithm_name', algorithm_names)
def test_new_optimizer_frechet(algorithm_name: str) -> None:
    """Test if the ASE3- and ASE4-style optimizers for `FrechetTarget`.

    At present, BFGS and LBGFS are tested.
    Within the history size, LBFGS should provide mathematically the same
    results as BFGS, and therefore we can use the same reference energy.
    """
    atoms = setup_surface()
    fmax = 0.01
    smax = 0.00001
    target = FrechetOptimizable(atoms, fmax=fmax, smax=smax)
    method = _make_algorithm_ase4(target, algorithm_name)
    opt = Optimizer(target, method)
    step = opt.run()

    assert step.gradient_obj.converged
    assert step.i == 18

    old_energy = old_frechet_energy(setup_surface(), fmax=fmax, smax=smax)
    assert target.get_value() == pytest.approx(old_energy)


@pytest.mark.parametrize('algorithm_name', algorithm_names)
def test_restart(tmp_path, algorithm_name) -> None:
    """Test if `Optimizer.restart` works as expected."""
    atoms = setup_surface()
    fmax = 0.001
    smax = 0.0001
    target = FrechetOptimizable(atoms, fmax=fmax, smax=smax)
    method = _make_algorithm_ase4(target, algorithm_name)

    restartpath = tmp_path / 'restart.json'
    trajpath = tmp_path / 'opt.traj'
    # trajpath.unlink(missing_ok=True)

    # Three use cases when starting a relaxation:
    #  * Wipe old files, start from scratch
    #  * Load from old files, and append (overwriting restartfile)
    #  * Load from old files, write to some other files
    opt = Optimizer(target, method, trajpath, restartpath)
    opt.run(steps=5)

    firstpart_images = read_images(trajpath)

    assert len(firstpart_images) == 6
    print(' --- first part done and saved, now continue ---')

    halfway = restartpath.with_name('halfway.json')
    write_restartfile(halfway, opt.method, opt.target, opt.step)

    step = opt.run()

    ref = pytest.approx(0.837190)
    gradient_obj = step.gradient_obj
    assert target.get_value() == ref
    assert gradient_obj.fnorm < fmax
    assert gradient_obj.snorm < smax
    assert step.i == 17

    images = read_images(trajpath)
    last_atoms = images[-1]

    assert len(images) == 18
    assert last_atoms.get_potential_energy() == ref

    print('done relaxing, now restart from checkpoint')
    lastpart_traj = tmp_path / 'lastpart.traj'

    opt = Optimizer.restart(halfway, EMT(), trajectory=lastpart_traj)
    opt.run()

    # If we want to (for example) "get the energy" or "get the calculator"
    # we will need to poke into the opt.target.xxx.  Maybe there should be
    # unified way to export the "domain stuff" similar to iterimages()

    lastpart_images = read_images(lastpart_traj)
    assert len(firstpart_images) + len(lastpart_images) == len(images)
    last_atoms2 = lastpart_images[-1]
    assert last_atoms2.get_potential_energy() == pytest.approx(ref)
    assert last_atoms.positions == pytest.approx(last_atoms2.positions)
    assert last_atoms.cell == pytest.approx(last_atoms2.cell)
