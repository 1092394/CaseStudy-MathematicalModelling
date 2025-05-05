import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sympy import zeros

from LevySearch import LevySearch

# Global variables, assign proper values before use
Gamma_n = None  # Relative decoherence rate for the two modules
Gamma_p = None  # Total decoherence rate for the two modules
U = None  # Conversion rate
Vm = None  # Amplitude of V
errorcounter = 0


def zeta_func(t, klist, omegalist):
    """
    Compute the zeta value using Fourier coefficients.
    Parameters:
        t : float
            Time.
        klist : array_like
            The first element is the constant term, and the rest are sine coefficients (a_n).
        omegalist : array_like
            The first element is the constant omega value used in sine and cosine, and the rest are cosine coefficients (b_n).
    Returns:
        result : float
            The computed zeta value.
    """
    if len(klist) != len(omegalist):
        raise ValueError("Your input klist must be of the same size as omegalist")
    # a0 = klist[0], omega = omegalist[0]
    alist = np.array(klist[1:])
    blist = np.array(omegalist[1:])
    N = len(blist)
    indices = np.arange(1, N + 1)
    result = klist[0] + np.sum(alist * np.sin(indices * omegalist[0] * t) + blist * np.cos(indices * omegalist[0] * t))
    return result


def compute_zetalist(zetatimelist, klist, omegalist):
    """
    Compute a list of zeta values for a given list of time points.
    Parameters:
        zetatimelist : array_like
            List of time points.
        klist : array_like
            Fourier coefficients for k.
        omegalist : array_like
            Fourier coefficients for omega.
    Returns:
        resultlist : ndarray
            An array of computed zeta values corresponding to the given time points.
    """
    resultlist = np.array([zeta_func(t, klist, omegalist) for t in zetatimelist])
    return resultlist


def MasterEquation(t, varvec, zetatimelist, zetalist):
    """
    Compute the system's master equation under the mean field method.
    Parameters:
        t : float
            Time.
        varvec : array_like, shape (3,)
            Variable vector in the order: [S, theta, n].
        zetatimelist : array_like
            List of time points for interpolation.
        zetalist : array_like
            Zeta values corresponding to the time points in zetatimelist.
    Returns:
        difvec : ndarray, shape (3,)
            Derivative vector in the order: [S', theta', n'].
    """
    global Gamma_n, Gamma_p, U, Vm
    # Get the zeta value at time t using linear interpolation
    zeta = np.interp(t, zetatimelist, zetalist)
    difvec = np.zeros(3)
    # Note: Python indexing starts at 0, so MATLAB's varvec(1) becomes varvec[0], etc.
    difvec[0] = (-2 * Vm * np.sqrt(varvec[2]) * (1 + varvec[0]) * np.sqrt(1 - varvec[0]) * np.sin(varvec[1])
                 - Gamma_n * (1 - varvec[0] ** 2))
    difvec[1] = (4 * U * varvec[2] * varvec[0] - zeta
                 - Vm * np.sqrt(varvec[2]) * (1 - 3 * varvec[0]) / np.sqrt(1 - varvec[0]) * np.cos(varvec[1]))
    difvec[2] = -(Gamma_p + Gamma_n * varvec[0]) * varvec[2]
    return difvec


def getstabletime(bound, S_vector, lower=True):
    """
    Find the index of the first time point in S_vector where the condition is continuously met.

    Parameters:
        bound : float
            The threshold value. For example, bound = 0.1 means the function searches for S <= 0.1 (if lower is True).
        S_vector : list or ndarray
            The sequence of S values.
        lower : bool, optional, default True
            If True, the condition is S <= bound; if False, the condition is S >= bound.

    Returns:
        The index (0-indexed) where the condition is continuously satisfied, or float('inf') if not.
    """
    result = float('inf')
    leng = len(S_vector)
    if not lower:
        vectorbool = [s >= bound for s in S_vector]
    else:
        vectorbool = [s <= bound for s in S_vector]
    for i in range(leng):
        if vectorbool[i]:
            if all(vectorbool[i:]):
                result = i
                break
    return result


def length_function(t_span, initvec, bound, zetatimelist, zetalist):
    """
    Compute the stable time of the current state, i.e., the time when the system reaches the "purified" state,
    which serves as the weight function for an ant colony algorithm.

    Parameters:
        t_span : tuple or list of length 2
            The time interval (start, end).
        initvec : list or ndarray of length 3
            Initial conditions for the master ODEs.
        bound : float
            The desired purification threshold.
        zetatimelist : list or ndarray
            Time points for interpolation.
        zetalist : list or ndarray
            Zeta values corresponding to the time points in zetatimelist.

    Returns:
        stable_time : float
            The time at which the system reaches a stable state; returns float('inf') if the state is not reached within the interval.
    """
    global Gamma_n, Gamma_p, U, errorcounter, Vm
    errorflag = False
    try:
        # Use the BDF method to solve the stiff ODE
        sol = solve_ivp(lambda t, varvec: MasterEquation(t, varvec, zetatimelist, zetalist),
                        t_span, initvec, method='BDF')
        if not sol.success:
            print("ODE solver error")
            errorcounter += 1
            errorflag = True
    except Exception as e:
        print("ODE solver error:", e)
        errorcounter += 1
        errorflag = True
    if not errorflag:
        # Get the sequence of S values from the solution
        S_vector = sol.y[0, :]
        stablestep = getstabletime(bound, S_vector)
        if stablestep != float('inf'):
            stable_time = sol.t[stablestep]
            # If the stable time is too close to the end of the time span, consider it as not stabilized.
            if stable_time >= 0.93 * t_span[1]:
                stable_time = float('inf')
        else:
            stable_time = float('inf')
    else:
        stable_time = float('inf')
    return stable_time


if __name__ == "__main__":
    plt.rcParams.update({
        'font.size': 16,  # 全局字体大小
        'axes.titlesize': 18,  # 子图标题字体大小
        'axes.labelsize': 20,  # 坐标轴标签字体大小
        'xtick.labelsize': 18,  # x轴刻度字体大小
        'ytick.labelsize': 18,  # y轴刻度字体大小
        'legend.fontsize': 18  # 图例字体大小
    })
    import time
    t1 = time.time()
    # Example: assign values to global variables
    Gamma_n = 0.2
    Gamma_p = 0.5
    U = 4
    Vm = 1.0
    errorcounter = 0

    # parameters
    end_time = 5
    t_span = (0, end_time)  # Time interval [0, 10]
    initvec = [0.99, 0.0, 1.0]  # Initial conditions [S, theta, n]
    bound = -0.9
    zetatimelist = np.linspace(0, end_time, 200)

    # Generate a Levy search net
    k_length = 10
    omega_length = 10

    searcher = LevySearch(
        dim=k_length + omega_length,
        epsilon=0.01,
        alpha=1.4,
        beta=0.7,
        K=6,  # archive bifurcation criterion
        scaling=0.1,
        seed=102,
        recency_ratio=0.5
    )

    start_point = np.ones(searcher.dim)
    steps = 10000
    search_r = 10
    perturb = 0.05  # 5% perturbation
    net_points = searcher.generate_searching_nodes(start_point=start_point, steps=steps, search_r=search_r)
    searching_points = np.array(net_points)



    stable_time_list = np.zeros(steps)
    for i in range(steps):
        pars = searching_points[i]
        klist = pars[0:k_length]  # The first element is the constant term, the rest are sine coefficients
        omegalist = pars[k_length:]  # The first element is the constant omega, the rest are cosine coefficients
        # Compute the list of zeta values
        zetalist_array = compute_zetalist(zetatimelist, klist, omegalist)

        # Compute the stable time
        stable_time = length_function(t_span, initvec, bound, zetatimelist, zetalist_array)
        # print("Stable time:", stable_time)
        stable_time_list[i] = stable_time

    min_time_idx = stable_time_list.argmin()
    # print(min_time_idx)
    # Del
    # min_time_idx = 500
    min_coef = searching_points[min_time_idx]
    klist = min_coef[0:k_length]  # The first element is the constant term, the rest are sine coefficients
    omegalist = min_coef[k_length:]  # The first element is the constant omega, the rest are cosine coefficients
    zetalist_array = compute_zetalist(zetatimelist, klist, omegalist)
    sol = solve_ivp(lambda t, varvec: MasterEquation(t, varvec, zetatimelist, zetalist_array),
                    t_span, initvec, method='BDF', dense_output=True)

    # min_stable_time_idx = getstabletime(bound, sol.y[0, :])
    # stable_time = sol.t[min_stable_time_idx]
    # stable_n = sol.y[2, min_stable_time_idx]
    t2 = time.time()

    print(t2 - t1)


    # Plot the evolution of S, theta, n and zeta
    plt.figure(figsize=(10, 8))
    # Plot S, theta, n evolution
    plt.subplot(2, 1, 1)
    plt.plot(sol.t, sol.y[0, :], label="S")
    # plt.plot(sol.t, sol.y[1, :], label="theta")
    plt.plot(sol.t, sol.y[2, :], label="n")
    plt.xlabel("Time")
    plt.ylabel("Variables")
    # plt.title(f"Evolution of S and n. t={stable_time}, n={stable_n}")
    plt.title(f"Evolution of S and n.")
    plt.legend()

    # Draw the stable time
    # plt.axvline(x=stable_time_list[min_time_idx], color="black", linestyle="--")
    plt.legend()

    # Draw the val = 0
    plt.axhline(y=0, color="black", linestyle="--")
    plt.legend()

    # Plot zeta evolution
    # Compute zeta values at the time points of the ODE solution
    zeta_sol = np.interp(sol.t, zetatimelist, zetalist_array)
    plt.subplot(2, 1, 2)
    plt.plot(sol.t, zeta_sol, label="zeta", color="red")
    plt.xlabel("Time")
    plt.ylabel("zeta")
    plt.title("Evolution of zeta")
    plt.legend()
    # Calculate 4UnS
    plt.plot(sol.t, 4 * U* sol.y[2, :] * sol.y[0, :], label="4UnS", color="blue")
    # Draw the stable time
    # plt.axvline(x=stable_time_list[min_time_idx], color="black", linestyle="--")
    plt.legend()

    # plt.subplot(3, 1, 3)
    # plt.plot(np.arange(1, steps + 1), stable_time_list)
    # plt.xlabel("Stable Time")
    # plt.ylabel("Steps")
    # plt.title("Stable time at each step")

    plt.tight_layout()
    plt.show()

    # plt.plot(searcher.get_archive_num_at_points())
    # plt.title("Local archive number at each steps")
    # plt.xlabel('Steps')
    # plt.ylabel('Archive Num')
    # plt.show()