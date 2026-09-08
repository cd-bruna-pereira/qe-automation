from dataclasses import asdict, dataclass
from typing import ClassVar

import numpy as np

from ase._4.optimize._symopt.relax import SymmetryAdaptedAtoms


@dataclass
class SymGradient:
    gradient: np.ndarray
    fnorm: float
    snorm: float
    converged: bool
    backF_av: np.ndarray
    backS_vv: np.ndarray
    symmetry_force_violation: float
    symmetry_stress_violation: float
    volume: float

    def loginfo(self):
        return {
            'F': self.fnorm,
            'S': self.snorm,
            'Ferr': self.symmetry_force_violation,
            'Serr': self.symmetry_stress_violation,
            'V': self.volume,
        }

    def datafy(self):
        return asdict(self)


@dataclass
class SymOpt:
    iotype: ClassVar[str] = 'symopt'

    symmatoms: SymmetryAdaptedAtoms
    fmax: float
    smax: float

    def step_to_string(self, step):
        return self.symmatoms.step_to_string(
            step.i, self.fmax, self.smax, step.gradient_obj
        )

    def log_headers(self):
        return self.symmatoms.logheaders()

    def gradient_norm(self, gradient):
        # Symmetry-reduced coordinates need their own norm,
        # not the Cartesian default.
        return self.symmatoms.gradient_norm(gradient)

    def datafy(self):
        # What do we need to save?
        # symmatoms defines the initial reduced positions as 0, so the
        # mapping absolutely needs saving aside from the Hessian.

        return {
            'fmax': self.fmax,
            'smax': self.smax,
            'symmatoms': self.symmatoms.datafy(),
        }

    @classmethod
    def undatafy(cls, dct, calc):
        dct = dct.copy()
        dct['symmatoms'] = SymmetryAdaptedAtoms.undatafy(dct['symmatoms'], calc)
        return cls(**dct)

    @classmethod
    def undatafy_gradient(cls, dct):
        return SymGradient(**dct)

    def get_value(self):
        return self.symmatoms.get_value()

    def get_gradient(self):
        return self.symmatoms._get_gradient(fmax=self.fmax, smax=self.smax)

    def get_x(self):
        return self.symmatoms.get_x()

    def set_x(self, x):
        self.symmatoms.set_x(x)

    def initial_hessian(self):
        raise NotImplementedError

    def iterimages(self):
        yield self.symmatoms.actual_atoms
