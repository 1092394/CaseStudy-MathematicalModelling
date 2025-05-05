import numpy as np
import random
from scipy.stats import levy_stable

class LevyFlight:
    """
    Define a Lévy flight class with parameters alpha, beta, scale and dim
    """

    def __init__(self, alpha: float, beta: float, scale: float, dim: int, seed:int=None):
        """
        :param alpha: float, (0<alpha<=2)
        :param beta: float, (0<beta<=1)
        :param scale: float, scale flight length (0<scale<=1)
        :param dim: int, dimension of flight
        :param seed: random seed
        """
        self.alpha = alpha
        self.beta = beta
        self.scale = scale
        self.dim = dim
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.path = None
        self.dir = None

    def generate_directions(self, num: int) -> np.ndarray:
        """
        Generate num uniformly distributed flight directions
        :return dir_list: numpy.ndarray [num, self.dim] like
        """
        dir_list = np.zeros((num, self.dim))
        for i in range(num):
            dir_list[i] = self._make_rand_unit_vector()
        self.dir = dir_list
        return dir_list

    def _make_rand_unit_vector(self):
        """
        Function source: https://stackoverflow.com/questions/6283080/random-unit-vector-in-multi-dimensional-space
        Based on the fact that if X = (X_1, ..., X_n) iid ~ N(0, 1), then X/sqrt(sum_i{X_i^2}) is ...
        uniformly distributed on the surface of unit sphere.
        """
        vec = np.random.normal(0.0, 1.0, size=self.dim)
        mag = np.linalg.norm(vec)
        return vec / mag

    def gen_flight(self, steps, start=None, directions=None):
        """
        Generate a flight.
        :param steps: int, number of  flight steps
        :param start: self.dim dimensional numpy ndarray, starting point of Lévy flight.
        :param directions: numpy.ndarray [num, self.dim]. If no directions are given, generate random directions under given seed.
        :return flight: numpy.ndarray [num, self.dim].
        """
        if directions is None:
            directions = self.generate_directions(steps)
        else:
            self.dir = directions
            if len(directions) < steps:
                raise ValueError(f"Not sufficient direction number. Current direction numbers {len(directions)}, {steps} steps needed.")


        self.path = np.zeros((steps, self.dim))
        if start is None:
            self.path[0] = np.zeros(self.dim)
        else:
            self.path[0] = start
            if len(start) != self.dim:
                raise ValueError(f"Wrong start point dimensions. Current flight dim {self.dim}, start point if of dim {len(start)}.")

        # Update path loc
        for i in range(1, steps):
            step_len = levy_stable.rvs(self.alpha, self.beta)
            self.path[i] = self.path[i - 1] + directions[i] * self.scale * step_len

        return self.path