import numpy as np

from .method import OptimizerMethod


def fd_hessian(target, eps=1e-5, tol=1e-8):
    xval_orig = target.get_x()

    ndofs = len(xval_orig)
    hessian = np.empty((ndofs, ndofs))

    xval = xval_orig.copy()

    for i in range(ndofs):
        xval[i] = xval_orig[i] + eps
        target.set_x(xval)
        gradplus = target.get_gradient().gradient
        xval[i] = xval_orig[i] - eps
        target.set_x(xval)
        gradminus = target.get_gradient().gradient
        xval[i] = xval_orig[i]

        dgrad_dx = (gradplus - gradminus) / (2 * eps)
        hessian[i, :] = dgrad_dx

    target.set_x(xval_orig)

    symm_error = abs(hessian.T - hessian).max()
    assert symm_error < tol, symm_error
    symm_hessian = 0.5 * (hessian.T + hessian)
    return symm_hessian


class BFGSMethod(OptimizerMethod):
    iotype = 'bfgs'
    methodname = 'BFGS'

    def __init__(self, hessian: np.ndarray):
        self.hessian = hessian

    def compute_step(self, gradient: np.ndarray) -> np.ndarray:
        omega, vectors = np.linalg.eigh(self.hessian)
        # Not sure what we should do about negative eigenvalues,
        # taking the absolute value is arbitrary and probably not good.
        return -vectors @ (gradient @ vectors / np.fabs(omega))

    def update(
        self,
        pos: np.ndarray,
        gradient: np.ndarray,
        pos0: np.ndarray,
        gradient0: np.ndarray,
    ) -> None:
        dpos = pos - pos0

        if np.abs(dpos).max() < 1e-7:
            # Same configuration again (maybe a restart).
            #
            # This happens when the class is used by the old code,
            # but it shouldn't generally trigger in ase._4.
            return

        dgradient = gradient - gradient0
        a = dpos @ dgradient
        dg = self.hessian @ dpos
        b = dpos @ dg
        self.hessian += (
            np.outer(dgradient, dgradient) / a - np.outer(dg, dg) / b
        )

    def datafy(self) -> list[float]:
        return self.hessian.ravel().tolist()

    @classmethod
    def undatafy(cls, hessian: list[float]):
        from math import isqrt

        n = isqrt(len(hessian))
        hessian_array = np.array(hessian).reshape(n, n)
        return cls(hessian_array)
