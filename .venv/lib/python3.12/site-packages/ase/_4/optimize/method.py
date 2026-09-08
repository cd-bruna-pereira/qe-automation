from abc import ABC, abstractmethod

import numpy as np


class OptimizerMethod(ABC):
    @abstractmethod
    def compute_step(self, gradient: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def update(
        self,
        pos: np.ndarray,
        gradient: np.ndarray,
        pos0: np.ndarray,
        gradient0: np.ndarray,
    ) -> None: ...

    @abstractmethod
    def datafy(self): ...

    @abstractmethod
    def undatafy(self, *args, **kwargs): ...
