import numpy as np
from scipy.sparse import coo_array

from .method import OptimizerMethod


class LBFGSMethod(OptimizerMethod):
    iotype = 'lbfgs'
    methodname = 'LBFGS'

    def __init__(
        self,
        *,
        memory: int = 100,
        initial_inverse_hessian: coo_array,
    ):
        self.memory = memory
        self.H0 = initial_inverse_hessian
        self.iteration = 0
        self.s: list[np.ndarray] = []
        self.y: list[np.ndarray] = []
        # Store also rho, to avoid calculating the dot product again and again.
        self.rho: list[float] = []

    def compute_step(self, gradient: np.ndarray) -> np.ndarray:
        loopmax = np.min([self.memory, self.iteration])
        a = np.empty(loopmax, dtype=np.float64)

        q = gradient.copy()
        # ## The algorithm itself:
        for i in range(loopmax - 1, -1, -1):
            a[i] = self.rho[i] * np.dot(self.s[i], q)
            q -= a[i] * self.y[i]
        z = self.H0 @ q

        for i in range(loopmax):
            b = self.rho[i] * np.dot(self.y[i], z)
            z += self.s[i] * (a[i] - b)

        self.iteration += 1

        return -z

    def update(
        self,
        pos: np.ndarray,
        gradient: np.ndarray,
        pos0: np.ndarray,
        gradient0: np.ndarray,
    ) -> None:
        s0 = pos - pos0
        self.s.append(s0)

        y0 = gradient - gradient0
        self.y.append(y0)

        rho0 = 1.0 / np.dot(y0, s0)
        self.rho.append(rho0)

        if self.iteration > self.memory:
            self.s.pop(0)
            self.y.pop(0)
            self.rho.pop(0)

    def datafy(self) -> dict:
        arr = self.H0
        rows = arr.row.tolist()
        cols = arr.col.tolist()
        vals = arr.data.tolist()
        d = {'data': list(zip(rows, cols, vals)), 'shape': arr.shape}
        return {
            'memory': self.memory,
            'initial_inverse_hessian': d,
            'shape': arr.shape,
            's': [_.tolist() for _ in self.s],
            'y': [_.tolist() for _ in self.y],
            'rho': np.array(self.rho).tolist(),
            'iteration': self.iteration,
        }

    @classmethod
    def undatafy(cls, method_data: dict):
        memory = method_data['memory']
        data = method_data['initial_inverse_hessian']['data']
        shape = method_data['initial_inverse_hessian']['shape']
        rows, cols, data = zip(*[(i, j, v) for i, j, v in data])
        arr = coo_array((data, (rows, cols)), shape=shape)
        self = cls(memory=memory, initial_inverse_hessian=arr)
        self.y = [np.array(_) for _ in method_data['y']]
        self.s = [np.array(_) for _ in method_data['s']]
        self.rho = method_data['rho']
        self.iteration = method_data['iteration']
        return self
