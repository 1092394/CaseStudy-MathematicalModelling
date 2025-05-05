import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import floor
import math

from LevySearch import LevySearch

def double_well_oracle(x, y):
    """
    This double well potential function have global minima at (x,y) = (\pm a, ~) and
    a local maxima at (x,y) = (a, ~)
    Programme is only allowed to ask values of this oracle
    """
    # parameters
    a = 150
    b = 10
    return (x ** 2 - a ** 2) ** 2 + b * y ** 2

def weierstrass_2d_approx(x, y, a=0.5, b=2, N=600):
    total = 0.0
    for n in range(N):
        term = (a**n) * np.cos((b**n) * np.pi * (np.sqrt(x**2 + y**2)))
        total += term
    return total

def f(x, y, a=0.5, b=2, N=20):
    """
    Returns f(x,y) = W2(x,y) + x^2 + y^2,
    where W2(x,y) is the (approx.) 2D Weierstrass function.
    """
    return weierstrass_2d_approx(x, y, a=a, b=b, N=N) + x*x + y*y

def nondiff_oracle(x):
    # temp = abs(x[0] - 1) + abs(x[1] - 2)

    return -(weierstrass_2d_approx(x[0], x[1]) - x[0]**2 - x[1]**2) + 2


def visualize_search_results(searcher, searching_points, function_values, levyminloc, otherminloc, search_r=np.inf):


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')


    grid_range = 1.7# search_r if search_r != np.inf else 300
    X = np.linspace(-grid_range, grid_range, 100)
    Y = np.linspace(-grid_range, grid_range, 100)
    X, Y = np.meshgrid(X, Y)
    # Z = double_well_oracle(X, Y)
    Z = nondiff_oracle([X, Y])
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3, rstride=1, cstride=1, linewidth=0, antialiased=True)


    ax.scatter(searching_points[:, 0], searching_points[:, 1], function_values,
               c='blue', s=2, label='Search Points', alpha=0.6)
    ax.scatter(levyminloc[0], levyminloc[1], nondiff_oracle([levyminloc[0], levyminloc[1]]),
               marker='*', c='red', s=40, label=f'Levy Min={nondiff_oracle([levyminloc[0], levyminloc[1]]):.2f}', alpha=1)
    ax.scatter(otherminloc[0], otherminloc[1], nondiff_oracle([otherminloc[0], otherminloc[1]]),
               marker='*', c='c', s=40, label=f'Powell Min={nondiff_oracle([otherminloc[0], otherminloc[1]]):.2f}', alpha=1)


    ax.set_title(f"Oracle Surface and Searching Points (Min at ({round(searching_points[np.argmin(function_values), 0], 2)}, {round(searching_points[np.argmin(function_values), 1], 2)}), actual: (0, 0))")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Function Value (Z)")

    if search_r != np.inf:
        u = np.linspace(0, 2 * np.pi, 100)
        x_boundary = search_r * np.cos(u)
        y_boundary = search_r * np.sin(u)
        z_boundary = np.zeros_like(x_boundary)
        ax.plot(x_boundary, y_boundary, z_boundary, color='blue', linestyle='-', alpha=0.5)


    ax.view_init(elev=20, azim=200)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from scipy.stats import expon  # Example: Exponential distribution
    import time
    import scipy

    t1 = time.time()
    searcher = LevySearch(
        dim=2,
        epsilon=0.001,
        alpha=1.4,
        beta=0.7,
        K=60,  # archive bifurcation criterion
        scaling=0.01,
        # seed=10,
        recency_ratio=0.6
    )

    start_point = [1/np.sqrt(2) - 0.1, 1/np.sqrt(2) - 0.001]
    steps = 10000
    search_r = 1.2

    perturb = 0.2  # 5% perturbation

    net_points = searcher.generate_searching_nodes(start_point=start_point, steps=steps, search_r=search_r)
    searching_points = np.array(net_points)
    # function_values = np.array([double_well_oracle(x, y) for x, y in searching_points])
    function_values = np.array([nondiff_oracle(x) for x in searching_points])
    min1 = min(function_values)
    t2 = time.time()
    print(t2-t1)
    levyminloc = searching_points[np.argmin(function_values) ,:]
    t3 = time.time()
    result = scipy.optimize.minimize(nondiff_oracle, np.array(start_point), method='Powell')
    t4 = time.time()
    print(t4-t3)
    print(result.x)


    visualize_search_results(searcher, searching_points, function_values, levyminloc, result.x, search_r)
    print(min1)
    print(result.fun)