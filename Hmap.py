import numpy
import numpy as np
from plotly.express import imshow
from seaborn import heatmap
import matplotlib.pyplot as plt

import seaborn as sns

from main import *


def H(x: numpy.ndarray, z: numpy.ndarray, alpha: float, n: int) -> float:
    if np.any((z < 0) | (z > 1)):
        raise ValueError('z must be between 0 and 1')
    k_vals = np.arange(1, n + 1)
    coeffs = np.sqrt(2) * k_vals ** (-alpha)

    return np.einsum('jk,zk->zj', x[:, :n], coeffs * np.cos(np.outer(z, k_vals) * np.pi))


def plot_fourier_heatmap(Hres,
                         xticks: int = 10,
                         yticks: int = 6,
                         xlabel: str = "Iteration (j)",
                         ylabel: str = r"$z \in [0,1]$",
                         title: str = r"Heatmap of Fourier Novelty Search in $L^2([0,1])$",
                         cmap: str = "RdBu_r") -> None:

    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(Hres, cmap=cmap, cbar_kws={'label': 'H(z)'})
    xtick_pos = np.linspace(0, Hres.shape[1] - 1, xticks).astype(int)
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_pos, rotation=0)

    ytick_pos = np.linspace(0, Hres.shape[0] - 1, yticks).astype(int)
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(np.round(np.linspace(0, 1, yticks), 2))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()
    plt.show()



# if __name__ == '__main__':
#     num_steps = 20000
#     alpha = 1.2
#     beta = 0.7
#     scale = 0.001
#     dim = 4
#     t_step = 1000
#
#     alphaH = 0.4
#
#     zsteps = 100
#
#
#     dirList = genNDimDir(num_steps, dim=dim)
#
#     flight = finLevyFlight(alpha, beta, scale, dim, dirList, num_steps)
#
#     zlist = np.linspace(0, 1, zsteps)
#     Hres = H(flight, zlist, alphaH, dim)
#
#     # plt.plot(flight[:, 0], flight[:, 1], linestyle="-", marker="o", markersize=2, alpha=0.7)
#     # plt.scatter(flight[0][0], flight[0][1], color="red", marker="o", label="Start")
#     # plt.scatter(flight[-1][0], flight[-1][1], color="blue", marker="x", label="End")
#     # plt.title(r"Lévy Flight $\alpha={}, \beta={}, steps={}$".format(alpha, beta, num_steps))
#     # plt.xlabel("X Position")
#     # plt.ylabel("Y Position")
#     # plt.legend()
#     # plt.grid()
#     # plt.show()
#
#     plt.figure(figsize=(10, 6))
#     ax = heatmap(Hres, cmap="RdBu_r", cbar_kws={'label': 'H(z)'})
#
#     ax.set_xticks(np.linspace(0, Hres.shape[1] - 1, 10).astype(int))  # x轴仅显示10个刻度
#     ax.set_yticks(np.linspace(0, Hres.shape[0] - 1, 6).astype(int))  # y轴仅显示6个刻度
#
#     ax.set_xticklabels(ax.get_xticks(), rotation=0)
#     ax.set_yticklabels(np.round(np.linspace(0, 1, 6), 2))  # z in [0,1]
#
#     plt.xlabel("Iteration (j)")
#     plt.ylabel(r"$z \in [0,1]$")
#     plt.title(r"Heatmap of Fourier Novelty Search in $L^2([0,1])$")
#     plt.tight_layout()
#     plt.show()



