from ase._4.pes import PotentialEnergySurface
from ase.utils.abc import Optimizable as Optimizable3


class OptimizablePES(Optimizable3):
    """Optimisation of potential energy surface (Atoms + Calculator)."""

    def __init__(
        self,
        pes: PotentialEnergySurface,
    ):
        self.pes = pes

    def get_x(self):
        return self.pes.get_positions().ravel()

    def set_x(self, x):
        self.pes.set_positions(x.reshape(-1, 3))

    def get_gradient(self):
        return -self.pes.get_property('forces').ravel()

    def get_value(self):
        if 'free_energy' in self.pes.requested_properties:
            prop_key = 'free_energy'
        else:
            prop_key = 'energy'
        return self.pes.get_property(prop_key)

    def iterimages(self):
        return self.pes.atoms.iterimages()

    def ndofs(self):
        return 3 * len(self.pes.atoms)
