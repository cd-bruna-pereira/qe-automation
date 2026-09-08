import pytest

from ase._4.optimize.optimize import OptimizablePES
from ase._4.pes import PotentialEnergySurface
from ase.optimize import BFGS


def _wrap_with_counter(calculator):
    """Wrap `calculator.evaluate` so calls are counted. Returns a `[n]` list."""
    counter = [0]
    original = calculator.evaluate

    def counting_evaluate(atoms, properties=None):
        counter[0] += 1
        return original(atoms, properties)

    calculator.evaluate = counting_evaluate
    return counter


def test_caches_within_same_state(v4atoms, calculator):
    """Repeated property lookups on a fresh PES share a single evaluate()."""
    counter = _wrap_with_counter(calculator)
    pes = PotentialEnergySurface(v4atoms, calculator)

    pes.get_property('forces')
    assert counter[0] == 1

    pes.get_property('energy')
    assert counter[0] == 1


def test_set_positions_invalidates(v4atoms, calculator):
    """set_positions invalidates the cache so the next evaluate() recomputes."""
    counter = _wrap_with_counter(calculator)
    pes = PotentialEnergySurface(v4atoms, calculator)

    pes.get_property('forces')
    assert counter[0] == 1

    pes.set_positions(pes.get_positions())
    pes.get_property('forces')
    assert counter[0] == 2


def test_setting_atoms_invalidates(v4atoms, calculator):
    """Replacing pes.atoms invalidates the cache so evaluate() recomputes."""
    counter = _wrap_with_counter(calculator)
    pes = PotentialEnergySurface(v4atoms, calculator)

    pes.get_property('forces')
    assert counter[0] == 1

    pes.atoms = v4atoms
    pes.get_property('forces')
    assert counter[0] == 2


def test_setting_calc_invalidates(v4atoms, calculator):
    """Replacing pes.calc invalidates the cache so evaluate() recomputes."""
    counter = _wrap_with_counter(calculator)
    pes = PotentialEnergySurface(v4atoms, calculator)

    pes.get_property('forces')
    assert counter[0] == 1

    pes.calc = calculator
    pes.get_property('forces')
    assert counter[0] == 2


def test_pes_rejects_unknown_requested_properties(v4atoms, calculator):
    """PES construction fails fast if a requested property is not
    supported by the calculator.
    """
    assert 'dipole' not in calculator.implemented_properties
    with pytest.raises(ValueError, match='dipole'):
        PotentialEnergySurface(
            v4atoms,
            calculator,
            requested_properties=('energy', 'forces', 'dipole'),
        )


def test_bfgs_one_evaluate_per_step(v4atoms, calculator):
    """A BFGS run should trigger exactly one evaluate() per optimizer step,
    not two (gradient + value were previously evaluated independently)."""
    v4atoms = v4atoms * (2, 2, 2)
    v4atoms.rattle(stdev=0.01, seed=42)

    counter = _wrap_with_counter(calculator)
    pes = PotentialEnergySurface(v4atoms, calculator)
    optimizable = OptimizablePES(pes)

    steps = 5
    with BFGS(optimizable, logfile=None) as opt:
        opt.run(fmax=1e-6, steps=steps)

    # BFGS's Dynamics.irun() evaluates once at start, then once per step.
    assert counter[0] == steps + 1
