from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_array

from ase import Atoms
from ase.optimize.optimize import OptimizableAtoms as OptimizableAtoms3


@dataclass
class ForceGradient:
    gradient: np.ndarray
    forces: np.ndarray
    fnorm: float
    converged: bool

    def loginfo(self) -> dict[str, float]:
        return {'fmax': self.fnorm}


class Optimizable4(ABC):
    @abstractmethod
    def get_value(self) -> float: ...

    @abstractmethod
    def get_gradient(self) -> ForceGradient: ...

    @abstractmethod
    def get_x(self) -> np.ndarray: ...

    @abstractmethod
    def set_x(self, x: np.ndarray) -> None: ...

    @abstractmethod
    def initial_hessian(self, *args, **kwargs) -> np.ndarray: ...

    @abstractmethod
    def initial_inverse_hessian(self, *args, **kwargs) -> coo_array: ...

    @abstractmethod
    def gradient_norm(self, gradient: np.ndarray) -> float: ...


class OptimizableAtoms(Optimizable4):
    def __init__(self, atoms: Atoms, fmax: float):
        self.optimizable: OptimizableAtoms3 = atoms.__ase_optimizable__()
        self.fmax = fmax

    def get_value(self) -> float:
        return self.optimizable.get_value()

    def get_gradient(self) -> ForceGradient:
        forces = self.optimizable.atoms.get_forces()
        gradient = -forces.ravel()
        fnorm = get_maxforce(forces)
        converged = fnorm < self.fmax
        return ForceGradient(
            gradient=gradient,
            forces=forces,
            fnorm=fnorm,
            converged=converged,
        )

    def get_x(self) -> np.ndarray:
        return self.optimizable.get_x()

    def set_x(self, x: np.ndarray) -> None:
        self.optimizable.set_x(x)

    def initial_hessian(self, alpha: float = 70.0) -> np.ndarray:
        return initial_position_hessian(self.optimizable.ndofs(), alpha)

    def initial_inverse_hessian(self, alpha: float = 70.0) -> coo_array:
        return initial_inverse_position_hessian(self.optimizable.ndofs(), alpha)

    def gradient_norm(self, gradient: np.ndarray) -> float:
        return self.optimizable.gradient_norm(gradient)


def get_maxforce(forces: np.ndarray) -> float:
    return np.linalg.norm(forces, axis=1).max()


def initial_position_hessian(ndofs: int, alpha: float = 70.0) -> np.ndarray:
    return np.diag(np.full(ndofs, 70.0))


def initial_inverse_position_hessian(
    ndofs: int,
    alpha: float = 70.0,
) -> coo_array:
    diags = np.full(ndofs, 1.0 / alpha)
    rows = cols = np.arange(ndofs)
    return coo_array((diags, (rows, cols)), shape=(ndofs, ndofs))
