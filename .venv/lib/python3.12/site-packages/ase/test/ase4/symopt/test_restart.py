import numpy as np
import pytest

from ase._4.optimize._symopt.relax import SymmetryAdaptedAtoms
from ase._4.optimize.bfgs import BFGSMethod
from ase._4.optimize.run import Optimizer
from ase._4.optimize.symopt import SymOpt
from ase.build import bulk
from ase.calculators.emt import EMT


def getatoms():
    atoms = bulk('Au', cubic=True)
    atoms.symbols = 'AuAgCuPt'
    # atoms.symbols[:2] = 'Ag'
    print(atoms)
    atoms.calc = EMT()
    return atoms


fmax = 1e-5
smax = 1e-7


def getopt(atoms, restartfile):
    symmatoms = SymmetryAdaptedAtoms.from_atoms_spglib(
        atoms, log=print, symprec=0.1
    )

    target = SymOpt(symmatoms, fmax=fmax, smax=smax)
    method = BFGSMethod(70.0 * np.identity(len(target.get_x())))

    return Optimizer(target, method, restartfile=restartfile)


def test_restart_consistent(tmp_path):
    restartfile = tmp_path / 'restart.json'

    atoms = getatoms()

    opt = getopt(atoms, restartfile)

    print('run 4')
    step4 = opt.run(steps=4)
    assert not step4.gradient_obj.converged
    assert step4.i == 4

    # This does not set e.g. the trajectory, restartfile, ...
    # It must of course be possible to set the restartfile through this.
    newopt = Optimizer.restart(restartfile, EMT())

    print('run +1')
    step5 = opt.run(steps=1)
    assert step5.i == 5
    assert not step5.gradient_obj.converged

    step5_restarted = newopt.run(steps=1)
    assert step5_restarted.i == 5
    assert not step5_restarted.gradient_obj.converged
    assert step5_restarted.gradient_obj.gradient == pytest.approx(
        step5.gradient_obj.gradient, abs=1e-15
    )

    # XXX restart and run one step.

    print('run +++')
    step11 = opt.run(steps=100)  # Converges at 11
    assert step11.gradient_obj.converged
    assert step11.i == 11

    # We are already converged so another run should run zero steps now:
    assert opt.run(steps=5).i == step11.i

    step11_restarted = newopt.run(steps=100)
    assert step11_restarted.gradient_obj.converged
    assert step11_restarted.i == 11
    assert step11_restarted.gradient_obj.gradient == pytest.approx(
        step11.gradient_obj.gradient, abs=1e-13
    )
