import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist, pdist, squareform

from Individual_implementation.TomVW.synthetic import composite_signal
from Individual_implementation.TomVW.timeObject import TimeObject


def hankel(signal: np.ndarray, m: int = 5):
    old_length = signal.shape[0]
    new_length = old_length - m + 1
    return scipy.linalg.hankel(signal[:new_length], signal[new_length-1:])


def hankel_martina(signal: np.ndarray, m: int = 5, t: int = 1):
    num_vectors = len(signal) - (m - 1) * t
    return np.array([signal[i:i + m*t:t] for i in range(num_vectors)])


def hankel_carl(signal: np.ndarray, m: int = 5, t: int = 1):
    old_length = signal.shape[0]
    new_length = old_length - m * t + 1

    hankel_like = np.zeros((new_length, m))  # Trajectory Matrix
    for i in range(new_length):
        hankel_like[i] = signal[i:i + m * t:t]


def hankel_chatgpt(signal: np.ndarray, m: int = 5, t: int = 1):
    n = len(signal)
    num_rows = n - (m - 1) * t
    # if num_rows <= 0:
    #     raise ValueError("Step size and m are too large for the length of the signal.")
    indices = np.arange(num_rows)[:, None] + np.arange(0, m*t, t)
    result = signal[indices]
    return result


def compare_hankel(n_samples: int = 10_000):
    # my_signal = composite_signal(n_samples, ((0.01, 4), (0.02, 2), (0.04, 1)))    # ((1, 4), (2, 2), (4, 1))
    funcs = [hankel,
             hankel_martina,
             hankel_carl,
             hankel_chatgpt]
    sizes = np.arange(100_000, 1_000_000, 100_000)
    signals = [composite_signal(size, ((0.01, 4), (0.02, 2), (0.04, 1))) for size in sizes]
    hankel_likes = []
    durations = []
    time_obj = TimeObject()
    # initialize hankel
    # scipy.linalg.hankel(np.array([1, 2, 3, 4]))

    for func in funcs:
        # hankel_likes.append(func(my_signal))
        # time_obj.new(f"{func.__name__}")
        durations2 = []
        for signal in signals:
            time_obj.new()
            hankel_likes.append(func(signal))
            durations2.append(time_obj.new())
        durations.append(durations2)

    fig, ax = plt.subplots()
    for i, func in enumerate(funcs):
        ax.plot(sizes, durations[i], label=f"{func.__name__}")
    ax.set_xlabel("Amount of samples (#)")
    ax.set_ylabel("Time (s)")
    ax.legend()
    plt.show()


def rp_cdist(hankel_like):
    return cdist(hankel_like, hankel_like, metric='euclidean')


def rp_pdist(hankel_like):
    return squareform(pdist(hankel_like, metric='euclidean'))


def compare_rp():
    min_size = 1_000
    max_size = 10_000
    sizes = np.arange(min_size, max_size, min_size)
    max_signal = composite_signal(max_size, ((0.01, 4), (0.02, 2), (0.04, 1)))
    max_hankel = hankel_chatgpt(max_signal, 5, 2)
    hankels = [max_hankel[:size] for size in sizes]
    funcs = [rp_cdist, rp_pdist]
    durations = []
    rps = []
    time_obj = TimeObject()
    for func in funcs:
        durations2 = []
        for current_hankel in hankels:
            time_obj.new()
            rps.append(func(current_hankel))
            durations2.append(time_obj.new())
        durations.append(durations2)
    time_obj.total()

    fig, ax = plt.subplots()
    for i, func in enumerate(funcs):
        ax.plot(sizes, durations[i], label=f"{func.__name__}")
    ax.set_xlabel("Amount of samples (#)")
    ax.set_ylabel("Time (s)")
    ax.legend()
    plt.show()


def test3(signal: np.ndarray, m: int = 5, t: int = 1):
    old_length = signal.shape[0]
    new_length = old_length - (m - 1) * t
    signal = signal.reshape(old_length, 1)
    my_distances = cdist(signal, signal, "euclidean")
    flat_distances = my_distances.reshape(1, old_length*old_length)
    # starting_indices = np.arange(new_length*old_length)
    # single_tile = np.ones(old_length)
    # single_tile[new_length:] = np.zeros((m - 1) * t)
    pattern_indices = np.arange(new_length)
    full_block_indices = np.concatenate([pattern_indices + i * old_length for i in range(new_length)])
    # full_block_indices = np.concatenate([np.arange(i*old_length, i*old_length + new_length) for i in range(new_length)])

    indices = full_block_indices[:, None] + np.arange(0, m*(old_length+1), old_length+1)
    my_view = flat_distances[indices]
    result = np.sum(my_view, axis=1)
    return result.reshape((new_length, new_length))


if __name__ == "__main__":
    compare_hankel()
    compare_rp()

