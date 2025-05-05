import AbstractBasis
import numpy as np

class FunctionCombiner:
    def __init__(self, basis: AbstractBasis, coefficients: np.ndarray = None):
        """
        :param basis: an object implemented AbstractBasis interface
        :param coefficients: a list with coefficients
        """
        self.basis = basis

        # Set coefficients to be 0s if they are not specified
        if coefficients is None:
            self.coefficients = np.zeros(self.basis.dim_basis())
        else:
            self.set_coefficients(coefficients)

    def set_coefficients(self, coeffs: np.ndarray):
        """
        Set coefficients
        """
        coeffs = np.array(coeffs, dtype=float)
        if coeffs.shape[0] != self.basis.dim_basis():
            raise ValueError(
                f"Length of coeffs ({coeffs.shape[0]}) != dim_basis ({self.basis.dim_basis()})"
            )
        self.coefficients = coeffs

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the value of this linear combination: f(x) = \sum_i c_i * basis(x)
        :param x: a numpy array, vector like, points to be evaluated
        :return: A numpy ndarray
        """
        return self.coefficients @ self.basis.evaluate_basis(x).transpose()

    def _H(self, z: np.ndarray, alpha: float) -> np.ndarray:
        """
        H generates the function of H map according to basis types
        """
        if np.any((z < 0) | (z > 1)):
            raise ValueError('z must be between 0 and 1')

        basisVal = self.basis.HBasis(z)
        k_vals = np.arange(1, self.basis.dim_basis() + 1)
        coeffs = k_vals ** (-alpha) # Adopt decay of higher term to ensure convergence

        return np.einsum('jk,zk->zj', self.coefficients[:, :self.basis.dim_basis()],
                         coeffs * basisVal)

    def heatmapData(self, alphaH = 0.5, zsteps = 100):
        zlist = np.linspace(0, 1, zsteps)
        return self._H(zlist, alphaH)

