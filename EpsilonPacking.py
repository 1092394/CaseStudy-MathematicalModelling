from holoviews.plotting.bokeh.styles import alpha
from scipy.stats import rv_continuous
import numpy as np
import scipy
import random

class EpsilonPacking:
    def __init__(
            self,
            dim,
            epsilon,
            distribution: rv_continuous,
            scaling,
            recency_ratio,
            seed=None
    ):
        self.dim = dim
        self.epsilon = epsilon
        self.distribution = distribution
        self.scaling = scaling
        self.recency_ratio = recency_ratio
        self.points = []
        self.seed = seed
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

    def _is_valid_candidate(self, candidate: np.ndarray) -> bool:
        temp = np.linalg.norm(candidate - self.points, axis=1) >= self.epsilon
        return all(np.linalg.norm(candidate - self.points, axis=1) >= self.epsilon)


    def generate_net(self, start_point, steps, children_per_step=3):
        """
        Generate an epsilon net.
        """
        self.points = [np.array(start_point)]
        ppp = [[np.array(start_point)]]

        for _ in range(steps):
            # Identify the subset of points for exploration (recency bias)
            recency_count = max(1, int(len(self.points) * self.recency_ratio))
            recent_points = self.points[-recency_count:]
            idx = np.random.randint(len(recent_points))
            parent = recent_points[idx]
            count = 0
            while count < children_per_step:
                p1 = self._generate_directions(1)[0]
                p2 = self.distribution.rvs()
                candidate = parent + p1 * p2 * self.scaling
                if self._is_valid_candidate(candidate):
                    count += 1
                    self.points.append(candidate)

            # for parent in recent_points:
                # count = 0
                # while count < children_per_step:
                #     p1 = self._generate_directions(1)[0]
                #     p2 = self.distribution.rvs()
                #     candidate = parent + p1 * p2 * self.scaling
                #     if self._is_valid_candidate(candidate):
                #         count += 1
                #         self.points.append(candidate)

        return self.points

    def print_net(self):
        """
        Print net nodes (one per line).
        """
        for pt in self.points:
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
        """
        Get the list/array of points in the net.
        """
        return np.array(self.points)

    def get_epsilon(self):
        return self.epsilon

    def get_dimension(self):
        return self.n_dim

    def get_recency_ratio(self):
        return self.recency_ratio

    def get_distribution(self):
        return self.distribution

    def get_points_num(self):
        return len(self.points)

    def vol_covered(self):
        return len(self.points) * (np.pi ** (self.dim / 2) * (self.epsilon/2) ** self.dim) / (scipy.special.gamma(self.dim / 2 + 1))




import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import expon  # Example: Exponential distribution
# #####################################################
# # Initialize the distribution
# # my_dist = scipy.stats.levy_stable(1.8, .5)
# my_dist = scipy.stats.norm()
# # Create an EpsilonPacking object
# packer = EpsilonPacking(
#     dim=2,
#     epsilon=1,
#     distribution=my_dist,
#     scaling=2,
#     recency_ratio=0.2,
# )
#
# # Generate the net
# net_points = packer.generate_net(start_point=[0, 0], steps=800, children_per_step=1)
#
# # Check if net is valid
# # print("Valid Net?", packer.check_valid_net())
# print("Total Points:", packer.get_points_num())
#
#
# # Plot the result (2D case) with SMALLER points and epsilon/2 circles
# points = np.array(net_points)
#
# fig, ax = plt.subplots(figsize=(8, 8))
#
# # Scatter plot with smaller points
# ax.scatter(points[:, 0], points[:, 1], c='blue', s=10, label='Net Points')
#
# # Draw a circle of radius epsilon/2 around each point
# for point in points:
#     circle = patches.Circle(
#         (point[0], point[1]),
#         radius=packer.get_epsilon() / 2,
#         fill=False,
#         edgecolor='red',
#         linestyle='--',
#         alpha=0.5
#     )
#     ax.add_patch(circle)
#
# # Plot settings
# ax.set_title("2D Epsilon Net with Epsilon/2 Circles")
# ax.set_xlabel("X-axis")
# ax.set_ylabel("Y-axis")
# ax.legend()
# ax.set_aspect('equal', 'box')
# plt.grid(True)
# plt.show()
# print(packer.vol_covered())
#

##########################################################################
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18
})

dim = 2
epsilon = 1
scaling = 2
recency_ratio = 0.2
steps = 800
children_per_step = 1
seed = 40

normal_dist = scipy.stats.norm()

levy_dist = scipy.stats.levy_stable(1.4, 0.7)

packer_normal = EpsilonPacking(dim, epsilon, normal_dist, scaling, recency_ratio, seed=seed)
net_points_normal = packer_normal.generate_net(start_point=[0, 0], steps=steps, children_per_step=children_per_step)

packer_levy = EpsilonPacking(dim, epsilon, levy_dist, scaling, recency_ratio, seed=seed)
net_points_levy = packer_levy.generate_net(start_point=[0, 0], steps=steps, children_per_step=children_per_step)

points_normal = np.array(net_points_normal)
points_levy = np.array(net_points_levy)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

def plot_net(ax, points, title):
    ax.scatter(points[:, 0], points[:, 1], c='blue', s=10, label='Net Points')
    for point in points:
        circle = patches.Circle(
            (point[0], point[1]),
            radius=epsilon / 2,
            fill=False,
            edgecolor='red',
            linestyle='--',
            alpha=0.5
        )
        ax.add_patch(circle)
    ax.set_title(title)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.legend()
    ax.set_aspect('equal', 'box')
    ax.grid(True)

plot_net(axes[0], points_normal, "Standard Normal Distribution Net")

plot_net(axes[1], points_levy, "Lévy Search Nodes")

plt.suptitle("Comparison of Epsilon Nets: Normal vs. Lévy Distribution", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("Normal Net Covered Volume:", packer_normal.vol_covered())
print("Lévy Net Covered Volume:", packer_levy.vol_covered())

