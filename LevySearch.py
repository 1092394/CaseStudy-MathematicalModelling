from re import search

import numpy as np
import scipy
from scipy.stats import rv_continuous
import random

class LevySearch:
    def __init__(
            self,
            dim,
            epsilon,
            alpha,
            beta,
            K, # archive bifurcation criterion
            scaling,
            recency_ratio,
            seed=None
    ):
        self.dim = dim
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.scaling = scaling
        self.recency_ratio = recency_ratio
        self.__distribution = scipy.stats.levy_stable(alpha=self.alpha, beta=self.beta)
        self.__points = []
        self.__archive: list = []
        self.__archive_num_at_points = [] # this property sorts archive num at each point, which should be monotonic increasing
        self.seed = seed
        self.start_point = []
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def _generate_directions(self, num: int) -> np.ndarray:
        """
        Generate num uniformly distributed flight directions
        :return dir_list: numpy.ndarray [num, self.dim] like
        """
        dir_list = np.zeros((num, self.dim))
        for i in range(num):
            dir_list[i] = self._make_rand_unit_vector()
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


    def _is_valid_candidate(self, candidate: np.ndarray, archive: np.ndarray) -> bool:
        return all(np.linalg.norm(candidate - archive, axis=1) >= self.epsilon)


    def generate_next_searching_node(self, neighbour_archive):
        """
        Generate the next searching node according to neighbour/local archive.
        Hopefully this will minimize searching costs
        :param neighbour_archive: numpy.ndarray like
        Assume the next step will only generate one node.
        """
        # Using local recency bias, recency ratio can be larger than global recency ratio
        recency_count = max(1, int(len(neighbour_archive) * self.recency_ratio))
        recent_points = neighbour_archive[-recency_count:]
        idx = np.random.randint(len(recent_points))
        parent = recent_points[idx]

        while True:
            p1 = self._generate_directions(1)[0]
            p2 = self.__distribution.rvs()
            candidate = parent + p1 * p2 * self.scaling
            if self._is_valid_candidate(candidate, neighbour_archive):
                return candidate, p2 * self.scaling


    def generate_searching_nodes(self, start_point, steps, search_r=np.inf):
        self.__points = [np.array(start_point)]
        self.__archive = [[np.array(start_point)]]
        self.start_point = np.array(start_point)
        for _ in range(steps):
            # Randomly choose an archive
            archive_num = len(self.__archive)
            archive_chosen_idx = np.random.randint(archive_num)

            flag = True
            while flag:
                next_point, step_length = self.generate_next_searching_node(self.__archive[archive_chosen_idx])
                if np.linalg.norm(next_point - start_point) <= search_r:
                    flag = False

            if abs(step_length) >= self.K * self.scaling:
                new_archive = [next_point]
                self.__archive.append(new_archive)
            else:
                self.__archive[archive_chosen_idx].append(next_point)

            self.__points.append(next_point)
            self.__archive_num_at_points.append(archive_num)

        return self.__points

    def print_net(self):
        """
        Print net nodes (one per line).
        """
        for pt in self.__points:
            print(" ".join(map(str, pt)))

    def check_valid_net(self):
        """
        Verify that the net is a valid epsilon-net (pairwise distances >= epsilon).

        Returns:
        --------
        bool
            True if valid, False otherwise.
        """
        if len(self.points) < 2:
            return True  # With <2 points, always valid
        arr = np.array(self.points)
        dists = scipy.spatial.distance.cdist(arr, arr)
        # For a valid epsilon-net, we need dists[i,j] >= epsilon for all i != j
        # We'll ignore the diagonal which is 0
        np.fill_diagonal(dists, np.inf)
        return np.all(dists >= self.epsilon)

    def get_points(self):
        return self.__points

    def get_epsilon(self):
        return self.epsilon

    def get_alpha(self):
        return self.alpha

    def get_beta(self):
        return self.beta

    def get_K(self):
        return self.K

    def get_scaling(self):
        return self.scaling

    def get_recency_ratio(self):
        return self.recency_ratio

    def vol_covered(self):
        return len(self.__points) * (np.pi ** (self.dim / 2) * (self.epsilon/2) ** self.dim) / (scipy.special.gamma(self.dim / 2 + 1))

    def get_archive(self):
        return self.__archive

    def get_points_num(self):
        return len(self.__points)

    def get_archive_num(self):
        return len(self.__archive)

    def get_archive_num_at_points(self):
        return self.__archive_num_at_points


########################################################################################################################
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from scipy.stats import expon  # Example: Exponential distribution

    # my_dist = scipy.stats.norm()
    # Create an EpsilonPacking object
    searcher = LevySearch(
        dim=2,
        epsilon=1,
        alpha=1.4,
        beta=0.7,
        K=60,  # archive bifurcation criterion
        scaling=1,
        seed=10,
        recency_ratio=0.3
    )

    start_point = [0, 0]
    steps = 5000
    search_r = np.inf

    # Generate the net
    net_points = searcher.generate_searching_nodes(start_point=start_point, steps=steps, search_r = search_r)

    # Check if net is valid
    # print("Valid Net?", packer.check_valid_net())
    print("Total Points:", searcher.get_points_num())

    # Plot the result (2D case) with SMALLER points and epsilon/2 circles
    points = np.array(net_points)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot with smaller points
    ax.scatter(points[:, 0], points[:, 1], c='blue', s=10, label='Net Points')

    # Draw a circle of radius epsilon/2 around each point
    for point in points:
        circle = patches.Circle(
            (point[0], point[1]),
            radius=searcher.get_epsilon() / 2,
            fill=False,
            edgecolor='red',
            linestyle='--',
            alpha=0.5
        )
        ax.add_patch(circle)

    # Plot settings
    ax.set_title("2D Epsilon Net with Epsilon/2 Circles")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.legend()
    ax.set_aspect('equal', 'box')
    # Draw the searching reign
    if search_r is not np.inf:
        ax.add_patch(patches.Circle((0, 0), search_r, color='b', fill=False))

    plt.grid(True)
    plt.show()
    print(searcher.vol_covered())

    print(searcher.get_points_num())
    print(searcher.get_archive_num())
    plt.plot(searcher.get_archive_num_at_points())
    plt.title("Local archive number at each steps")
    plt.xlabel('Steps')
    plt.ylabel('Archive Num')
    plt.show()
