import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.spatial.distance import cdist, pdist, squareform

from Individual_implementation.TomVW.recurrence import view_cdist
from Individual_implementation.TomVW.synthetic import composite_signal
from Individual_implementation.TomVW.timeObject import TimeObject


def hankel(signal: np.ndarray, m: int = 5):
    old_length = signal.shape[0]
    new_length = old_length - m + 1
    return scipy.linalg.hankel(signal[:new_length], signal[new_length-1:])


def hankel_list_comprehension(signal: np.ndarray, m: int = 5, t: int = 1):
    num_vectors = len(signal) - (m - 1) * t
    return np.array([signal[i:i + m*t:t] for i in range(num_vectors)])


def hankel_fill_zeros(signal: np.ndarray, m: int = 5, t: int = 1):
    new_length = signal.shape[0] - m * t + 1

    hankel_like = np.zeros((new_length, m))  # Trajectory Matrix
    for i in range(new_length):
        hankel_like[i] = signal[i:i + m * t:t]
    return hankel_like


def hankel_view(signal: np.ndarray, m: int = 5, t: int = 1):
    num_rows = len(signal) - (m - 1) * t
    indices = np.arange(num_rows)[:, None] + np.arange(0, m*t, t)
    return signal[indices]


def hankel_stride(signal: np.ndarray, m: int = 5, t: int = 1):
    num_rows = len(signal) - (m - 1) * t
    return np.lib.stride_tricks.as_strided(signal,
                                           shape=(num_rows, m),
                                           strides=(signal.strides[0], t*signal.strides[0]))


def hankel_stride_contiguous(signal: np.ndarray, m: int = 5, t: int = 1):
    num_rows = len(signal) - (m - 1) * t
    stride = np.lib.stride_tricks.as_strided(signal,
                                             shape=(num_rows, m),
                                             strides=(signal.strides[0], t * signal.strides[0]))
    return np.ascontiguousarray(stride)


def compare_hankel(max_size: int = 2_000_001):
    time_obj = TimeObject()
    fast_funcs = [hankel_view,
                  hankel,
                  hankel_stride_contiguous,
                  hankel_stride]
    slow_funcs = [hankel_list_comprehension,
                  hankel_fill_zeros]
    sizes = np.arange(100_000, max_size, 100_000)
    max_signal = composite_signal(max_size, ((0.001, 4), (0.002, 2), (0.004, 1)))
    durations = []
    time_obj.new("Setup")

    for func in fast_funcs:
        durations2 = []
        for size in sizes:
            func(max_signal[:size])
            durations2.append(time_obj.new(func.__name__))
        durations.append(durations2)

    for func in slow_funcs:
        durations2 = []
        for size in sizes[:4]:
            func(max_signal[:size])
            durations2.append(time_obj.new(func.__name__))
        durations.append(durations2)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)

    for i, func in enumerate(slow_funcs):
        ax.plot(sizes[:4], durations[i+4], label=f"{func.__name__}")
    for i, func in enumerate(fast_funcs):
        ax.plot(sizes, durations[i], label=f"{func.__name__}")

    formatter = ScalarFormatter()
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    ax.xaxis.set_major_formatter(formatter)

    ax.set_xlabel("Amount of samples (#)")
    ax.set_ylabel("Time (s)")
    ax.legend()

    plt.show()


def rp_cdist(hankel_like):
    return cdist(hankel_like, hankel_like, metric='euclidean')


def rp_pdist(hankel_like):
    return squareform(pdist(hankel_like, metric='euclidean'))


def double_for_loop(hankel_like: np.ndarray) -> np.ndarray:
    """ dfl: double for loop"""
    new_length = hankel_like.shape[0]
    ans = np.zeros((new_length, new_length))
    for i in range(new_length):
        for j in range(new_length):
            ans[i, j] = np.linalg.norm(hankel_like[i] - hankel_like[j])
    return ans


def compare_rp():
    time_obj = TimeObject()
    min_size = 500
    max_size = 10_000
    sizes = np.arange(min_size, max_size, min_size)
    max_signal = composite_signal(max_size, ((0.01, 4), (0.02, 2), (0.04, 1)))
    max_hankel = hankel_view(max_signal, 5, 2)
    funcs = [double_for_loop, rp_pdist, rp_cdist]
    durations = []
    time_obj.new("Setup")
    for func in funcs:
        durations2 = []
        for size in sizes:
            func(max_hankel[:size])
            durations2.append(time_obj.new(func.__name__))
            if (double_for_loop == func) and size >= 1_000:  # bad I know
                break
        durations.append(durations2)

    fig, ax = plt.subplots(dpi=200)
    for i, func in enumerate(funcs[1:]):
        ax.plot(sizes, durations[i + 1], label=f"{func.__name__}")
    ax.plot(sizes[:2], durations[0], label=f"{double_for_loop.__name__}")
    ax.set_xlabel("Amount of samples (#)")
    ax.set_ylabel("Time (s)")
    ax.legend()
    plt.show()


def threshold(distance_matrix: np.ndarray, epsilon: float = 0.1):
    return distance_matrix < epsilon


def threshold_cast(distance_matrix: np.ndarray, epsilon: float = 0.1):
    return (distance_matrix < epsilon).astype(int)


def threshold_ip(distance_matrix: np.ndarray, epsilon: float = 0.1):
    """ ip stands for in place, shouldn't need the return """
    distance_matrix[distance_matrix < epsilon] = 0
    distance_matrix[distance_matrix >= epsilon] = 1
    return distance_matrix


def threshold_ip2(distance_matrix: np.ndarray, epsilon: float = 0.1):
    """ ip stands for in place, shouldn't need the return """
    indices = distance_matrix < epsilon
    distance_matrix[indices] = 0
    distance_matrix[~indices] = 1
    return distance_matrix


def compare_threshold(max_size: int = 10_001):
    time_obj = TimeObject()
    my_signal = composite_signal(max_size, ((0.001, 4), (0.002, 2), (0.004, 1)))
    step_size = 500
    sizes = np.arange(step_size, max_size, step_size)
    funcs = [threshold_ip,
             threshold_cast,
             threshold,
             threshold_ip2]
    n_funcs = len(funcs)
    time_obj.new("setup")
    rp = view_cdist(my_signal)
    time_obj.new("RP")
    for size in sizes:
        for func in funcs:
            func(rp[:size])
            time_obj.new(func.__name__)

    fig, ax = plt.subplots(dpi=200)

    for i, func in enumerate(funcs):
        ax.plot(sizes, time_obj.time_list[2+i::n_funcs], label=f"{func.__name__}")
    ax.set_xlabel("Amount of samples (#)")
    ax.set_ylabel("Time (s)")
    ax.legend()
    time_obj.new("Plotting")
    plt.show()


if __name__ == "__main__":
    # compare_hankel()
    # compare_rp()
    compare_threshold()

