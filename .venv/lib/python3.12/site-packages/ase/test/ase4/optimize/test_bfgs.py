from ase._4.optimize.optimize import OptimizablePES
from ase._4.pes import PotentialEnergySurface
from ase.optimize import BFGS


def test_bfgs_v4(v4atoms, calculator):
    """Run BFGS optimization with v4 atoms + calculator."""

    v4atoms.rattle(stdev=0.01, seed=42)

    pes = PotentialEnergySurface(v4atoms, calculator)
    optimizable = OptimizablePES(pes)

    with BFGS(optimizable, logfile=None) as opt:
        converged = opt.run(fmax=0.05, steps=100)

    assert converged
