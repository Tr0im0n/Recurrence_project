import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.integrate import odeint

from Individual_implementation.Martina.lorenz import get_xyzs_lorenz
from Individual_implementation.TomVW.synthetic import sine, noise
from Individual_implementation.TomVW.recurrence import view_cdist_threshold, cross_recurrence


def thomas_attractor(state, b):
    x, y, z = state
    dxdt = np.sin(y) - b * x
    dydt = np.sin(z) - b * y
    dzdt = np.sin(x) - b * z
    return np.array([dxdt, dydt, dzdt])


def solve_thomas_attractor(num_steps, dt, b, ic):
    xyzs = np.empty((num_steps + 1, 3), dtype=float)
    xyzs[0] = ic
    for i in range(num_steps):
        xyzs[i + 1] = xyzs[i] + thomas_attractor(xyzs[i], b) * dt
    return xyzs


def fig1():
    fig, axs = plt.subplots(2, 4, height_ratios=(1/3, 1), figsize=(15, 5), dpi=100)
    (ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8) = axs
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.1)

    signal_length = 2_100
    show_length = 2001
    xs = np.arange(signal_length)
    xsmall = xs/100

    y_sine = sine(1/200, 1, xs)
    rp_sine = view_cdist_threshold(y_sine, 40, 2, 0.05)

    y_noise = noise(signal_length, noise_type="normal")
    rp_noise = view_cdist_threshold(y_noise, epsilon=0.38)

    xyz_lorenz = get_xyzs_lorenz(signal_length, 0.01, False)
    y_lorenz = xyz_lorenz[1:, 1]
    rp_lorenz = view_cdist_threshold(y_lorenz)

    # Parameters
    initial_state = [1.0, 0.0, 1.0]  # Initial conditions
    b = 0.2
    dt = 0.01
    supersampling = 15
    trajectory = solve_thomas_attractor(signal_length*supersampling, dt, b, initial_state)
    x_thomas = trajectory[1::supersampling, 0]
    rp_thomas = view_cdist_threshold(np.array(x_thomas))

    # fig = plt.figure(figsize=(10, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], lw=0.5)

    ax1.plot(xs[:show_length], y_sine[:show_length])
    ax1.set_title("Sine Wave")
    ax5.imshow(rp_sine[:show_length, :show_length], cmap="binary", origin="lower")

    ax2.plot(xs[:show_length], y_noise[:show_length])
    ax2.set_title("White Noise")
    ax6.imshow(rp_noise[:show_length, :show_length], cmap="binary", origin="lower")

    ax3.plot(xs[:show_length], y_lorenz[:show_length])
    ax3.set_title("Lorenz Series")
    ax7.imshow(rp_lorenz[:show_length, :show_length], cmap="binary", origin="lower")

    ax4.plot(xs[:show_length], x_thomas[:show_length])
    ax4.set_title("Thomas Series")
    ax8.imshow(rp_thomas[:show_length, :show_length], cmap="binary", origin="lower")

    formatter = ScalarFormatter()
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))

    for ax in (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8):
        ax.xaxis.set_major_formatter(formatter)

    for ax in (ax5, ax6, ax7, ax8):
        ax.yaxis.set_major_formatter(formatter)
        ax.set_xlabel("Time")
        ax.set_ylabel("Time")

    plt.show()


def autoregressive_noise(n: int = 1000, phi: float = 0.8, mu: float = 0,
                         sigma: float = 5, seed: int = 0) -> np.ndarray:
    np.random.seed(seed)  # For reproducibility
    my_noise = np.random.normal(mu, sigma, n)
    ar_noise = np.zeros(n)

    for i in range(1, n):
        ar_noise[i] = phi * ar_noise[i - 1] + my_noise[i]

    return ar_noise


def standardize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std


def fig2():
    fig, ax = plt.subplots(1, 1, dpi=150)

    signal_length = 1_006

    xyz_lorenz = get_xyzs_lorenz(signal_length, 0.02, False)
    y_lorenz = xyz_lorenz[1:, 1]
    std_lorenz = standardize(y_lorenz)

    ar_noise = autoregressive_noise(signal_length, seed=1)
    std_ar_noise = standardize(ar_noise)

    crp = cross_recurrence(std_ar_noise, std_lorenz)

    ax.imshow(crp, cmap="binary", origin="lower")
    ax.set_xlabel("Lorenz Index")
    ax.set_ylabel("AR Noise Index")
    plt.show()


def fig3():
    fig, ax = plt.subplots(1, 1, dpi=150)

    signal_length = 1_006

    xyz_lorenz = get_xyzs_lorenz(signal_length, 0.02, False)
    y_lorenz = xyz_lorenz[1:, 1]
    rp_lorenz = view_cdist_threshold(y_lorenz, epsilon=0.2)

    ar_noise = autoregressive_noise(signal_length, seed=1)
    rp_ar_noise = view_cdist_threshold(ar_noise, epsilon=0.2)

    jrp = rp_lorenz & rp_ar_noise

    ax.imshow(jrp, cmap="binary", origin="lower")
    ax.set_xlabel("Time")
    ax.set_ylabel("Time")
    plt.show()


if __name__ == "__main__":
    fig1()

