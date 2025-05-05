from abc import ABC, abstractmethod
import numpy as np

class AbstractBasis(ABC):
    @abstractmethod
    def evaluate_basis(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the value of basis for a given x
        :param x: ndarray like, points to be evaluated
        :return: ndarray like, shape = (len(x), num_basis), values at x for each basis
        """
        pass

    @abstractmethod
    def dim_basis(self) -> int:
        """
        :return: the dimension of the basis
        """
        pass

    @abstractmethod
    def HBasis(self, x: np.ndarray) -> np.ndarray:
        """
        :return: ndarray like, gives the orthogonal basis values
        """
        pass