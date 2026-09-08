import pytest

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.verlet import VelocityVerlet
from ase.units import fs


@pytest.fixture(name='atoms')
def fixture_atoms():
    atoms = bulk('Cu') * (2, 2, 2)
    atoms.calc = EMT()
    return atoms


def test_irun_call_order_through_md(atoms):
    """Unit test to fix the order in which functions
    that consume force/energy/stress evaluations are called.
    A safe-guard for ASEv4 developments.
    """
    md = VelocityVerlet(atoms, timestep=1.0 * fs)
    calls: list[str] = []

    def record(name):
        """Record the name instead of actually running
        the function.
        """

        def _recorder(*args, **kwargs):
            calls.append(name)

        return _recorder

    md.step = record('step')
    md._refresh_properties = record('_refresh_properties')
    md.log = record('log')
    md.call_observers = record('call_observers')

    list(md.irun(steps=2))

    assert calls == [
        '_refresh_properties',
        'log',
        'call_observers',  # init
        'step',
        '_refresh_properties',
        'log',
        'call_observers',  # iter 1
        'step',
        '_refresh_properties',
        'log',
        'call_observers',  # iter 2
    ]
