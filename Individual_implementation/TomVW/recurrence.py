
import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib.widgets import Slider, Button
from pyts.image import RecurrencePlot
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.signal import convolve2d

from Individual_implementation.TomVW.synthetic import composite_signal
from Individual_implementation.TomVW.timeObject import TimeObject


def get_length_matrix(signal: np.ndarray):
    """
    Just used in following function
    Just replace with squareform(pdist(...))
    Nvm use cdist(...)
    """
    length = signal.shape[0]
    ans = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            my_len = pow(signal[i] - signal[j], 2)
            ans[i, j] = my_len
    return ans


def double_for_loop_length_matrix(signal: np.ndarray, m: int = 5, t: int = 1):
    """ Double for loop implementation """
    length = signal.shape[0]
    signal = signal.reshape(length, 1)
    length_matrix = cdist(signal, signal)
    ans = np.zeros_like(length_matrix)

    for i in range(length-m):
        for j in range(length-m):
            # my_len = sum([length_matrix[i+k, j+k] for k in range(m)])
            # clip = 0 if my_len < 0.1 else 1
            ans[i, j] = np.sum(length_matrix[i:i + m * t:t, j:j + m * t:t])  # clip, my_len
    return ans


def double_for_loop_hankel(signal: np.ndarray, m: int = 5):
    """ Double for loop implementation """
    old_length = signal.shape[0]
    new_length = old_length - m + 1
    my_hankel = scipy.linalg.hankel(signal[:new_length], signal[new_length-1:])
    ans = np.zeros((new_length, new_length))

    for i in range(new_length):
        for j in range(new_length):
            ans[i, j] = np.linalg.norm(my_hankel[i] - my_hankel[j])
    return ans


def hankel_pdist(signal: np.ndarray, m: int = 5):
    """
    :param signal:
    :param m: amount of points we look at
    :return:
    """
    my_length = signal.shape[0]
    my_hankel = scipy.linalg.hankel(signal[:my_length - m + 1], signal[my_length - m:])
    return squareform(pdist(my_hankel, metric='euclidean'))


def hankel_kron_norm(signal: np.ndarray, m: int = 5):
    """
    :param signal:
    :param m: amount of points we look at
    :return:
    """
    old_length = signal.shape[0]
    new_length = old_length - m + 1
    my_hankel = scipy.linalg.hankel(signal[:new_length], signal[new_length-1:])
    my_ones = np.ones((new_length, 1))
    whatsthis = np.kron(my_ones, my_hankel) - np.kron(my_hankel, my_ones)
    row_norms = np.linalg.norm(whatsthis, axis=1)
    return row_norms.reshape(new_length, new_length)


def convolve_triangle_shift(signal: np.ndarray, m: int = 5):
    """
    Trying to implement my own recurrence plot.
    :param signal: np array of the signal
    :param m: The embedding dimension
    :return: The Recurrence plot, clipped?
    """
    my_length = signal.shape[0]
    signal = signal.reshape(my_length, 1)
    my_distances = pdist(signal, "euclidean")
    my_zeros = np.zeros((my_length, my_length))
    rows, cols = np.triu_indices(my_length, k=1)
    my_zeros[rows, cols] = my_distances
    flattened = my_zeros.reshape((1, my_length * my_length))
    upper_left_triangle = flattened[:, :-1].reshape((my_length-1, my_length+1))
    kernel = np.ones((m, 1))
    convolved = convolve2d(upper_left_triangle, kernel, "valid")
    flattened = convolved.reshape(1, (my_length-m)*(my_length+1))
    padded = np.zeros(my_length*my_length)
    padded[:(my_length-m)*(my_length+1)] = flattened
    half = padded.reshape((my_length, my_length))
    return half + half.T


def convolve_triangle_shift_with_timing(signal: np.ndarray, m: int = 5):
    """
    Trying to implement my own recurrence plot.
    :param signal: np array of the signal
    :param m: The embedding dimension
    :return: The Recurrence plot, clipped?
    """
    time_obj = TimeObject()
    my_length = signal.shape[0]
    signal = signal.reshape(my_length, 1)
    time_obj.new("Reshaping signal")
    my_distances = pdist(signal, "euclidean")
    time_obj.new("pdist")
    my_zeros = np.zeros((my_length, my_length))
    rows, cols = np.triu_indices(my_length, k=1)
    my_zeros[rows, cols] = my_distances
    time_obj.new("Triangle filling")
    flattened = my_zeros.reshape((1, my_length * my_length))
    upper_left_triangle = flattened[:-1].reshape((my_length-1, my_length+1))
    kernel = np.ones((m, 1))
    time_obj.new("Reshape Triangle")
    convolved = convolve2d(upper_left_triangle, kernel, "valid")
    time_obj.new("Convolution")
    flattened = convolved.reshape(1, (my_length-m)*(my_length+1))
    padded = np.zeros(my_length*my_length)
    padded[:(my_length-m)*(my_length+1)] = flattened
    half = padded.reshape((my_length, my_length))
    time_obj.new("Reshape")
    ans = half + half.T
    time_obj.new("Add 2 halves")
    time_obj.total()
    return ans


def convolve_diagonal(signal: np.ndarray, m: int = 5):
    """
    Trying to implement my own recurrence plot.
    :param signal: np array of the signal
    :param m: The embedding dimension
    :return: The Recurrence plot, clipped?
    """
    my_length = signal.shape[0]
    signal = signal.reshape((my_length, 1))
    distance_matrix = squareform(pdist(signal, "euclidean"))
    kernel = np.eye(m, dtype=int)
    return convolve2d(distance_matrix, kernel)


def martina(signal, m: int = 5, t: int = 1):
    num_vectors = len(signal) - (m - 1) * t
    vectors = np.array([signal[i:i + m*t:t] for i in range(num_vectors)])
    return squareform(pdist(vectors, metric='euclidean'))


def carl(signal, m: int = 5, t: int = 1, epsilon: float = 0.1):
    old_length = signal.shape[0]
    new_length = old_length - m*t + 1
    ones = np.ones((new_length, 1))

    hankel_like = np.zeros((new_length, m))    # Trajectory Matrix
    for i in range(new_length):
        hankel_like[i] = signal[i:i+m*t:t]

    whatsthis = np.kron(ones, hankel_like) - np.kron(hankel_like, ones)
    row_norms = np.linalg.norm(whatsthis, axis=1)
    return row_norms.reshape(new_length, new_length)

    # return squareform(pdist(P, 'euclidean'))    # distance_matrix =

    # recurrence_matrix = (distance_matrix <= epsilon).astype(int)
    # return recurrence_matrix


def test3(signal: np.ndarray, m: int = 5, t: int = 1):
    old_length = signal.shape[0]
    new_length = old_length - (m - 1) * t
    signal = signal.reshape((old_length, 1))
    my_distances = cdist(signal, signal, "euclidean")   # old_length x old_length
    flat_distances = my_distances.reshape(-1)   # 1, old_length*old_length
    pattern_indices = np.arange(new_length)
    full_block_indices = np.concatenate([pattern_indices + i * old_length for i in range(new_length)])  # new_length x new_length
    # full_block_indices = np.concatenate([np.arange(i*old_length, i*old_length + new_length) for i in range(new_length)])
    indices = full_block_indices[:, None] + np.arange(0, m*(old_length+1), old_length+1)
    my_view = flat_distances[indices]
    result = np.sum(my_view, axis=1)
    return result.reshape((new_length, new_length))


def package(signal: np.ndarray, m: int = 5, t: int = 1):
    time_obj = TimeObject()
    signal = signal.reshape((1, -1))
    time_obj.new("Reshape")
    rp = RecurrencePlot(m, t)
    time_obj.new("Init")
    ans = rp.transform(signal)[0]
    time_obj.new("Transform")
    return ans


def view_cdist(signal: np.ndarray, m: int = 5, t: int = 1):
    new_shape = signal.shape[0] - (m - 1) * t
    indices = np.arange(new_shape)[:, None] + np.arange(0, m * t, t)    # new_shape x m
    result = signal[indices]    # just a view
    return cdist(result, result, metric='euclidean')    # new_shape x new_shape


def stride_cdist(signal: np.ndarray, m: int = 5, t: int = 1):
    new_shape = signal.shape[0] - (m - 1) * t
    result = np.lib.stride_tricks.as_strided(signal,
                                             shape=(new_shape, m),
                                             strides=(signal.strides[0], signal.strides[0] * t))
    return cdist(result, result, metric='euclidean')


def epsilon_slider():
    time_obj = TimeObject()
    my_signal = create_signal1()
    time_obj.new()
    # my_length_matrix = get_length_matrix(my_signal)
    # time_obj.new()
    # my_recurrence_matrix = recurrence_matrix_slow(my_length_matrix)
    my_recurrence_matrix = fast(my_signal)
    time_obj.new()
    time_obj.total()

    plt.imshow(my_recurrence_matrix)
    plt.show()
    return
    # my_epsilon = 80
    # my_r = (my_recurrence_matrix <= my_epsilon).astype(int)
    # time_obj.new()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(np.arange(0, 10, 0.01), my_signal)
    ax1.set_title("My signal")
    ax2.imshow(my_recurrence_matrix, cmap="gray", origin="lower")   # [:990, :990]
    ax2.set_title("Clipped recurrence plot")

    def test1(epsilon):
        r = my_recurrence_matrix <= epsilon
        ax2.clear()
        ax2.imshow(r, cmap="gray_r", origin="lower")
        ax2.set_title("Clipped recurrence plot")
        plt.draw()

    epsilon_slider = Slider(plt.axes((0.55, 0.05, 0.4, 0.1)), "Epsilon", 1, 20, valinit=4, valstep=0.1)
    epsilon_slider.on_changed(test1)

    def show_distance(event):
        ax2.clear()
        ax2.imshow(my_recurrence_matrix, cmap="gray", origin="lower")   # [:990, :990]
        ax2.set_title("Clipped recurrence plot")
        plt.draw()

    distance_button = Button(plt.axes((0.55, 0.94, 0.1, 0.05)), "Distance")
    distance_button.on_clicked(show_distance)

    plt.show()


def compare_all(n_samples: int = 4_000, m: int = 5):
    time_obj = TimeObject()
    my_signal = composite_signal(n_samples, ((0.01, 4), (0.02, 2), (0.04, 1)))    # ((1, 4), (2, 2), (4, 1))
    funcs = [view_cdist,
             # hankel_pdist,
             # martina,
             # hankel_kron_norm,
             # double_for_loop_hankel,
             # double_for_loop_length_matrix,
             # convolve_triangle_shift,
             # convolve_diagonal,
             # carl,
             # test3,
             # package,
             stride_cdist]
    rps = []
    time_obj.new("Setup")
    for func in funcs:
        rps.append(func(my_signal, m))
        time_obj.new(f"{func.__name__}")

    return

    durations = time_obj.time_list[-2:]

    fig, axs = plt.subplots(1, 2)
    fig.suptitle(f"Samples: {n_samples}")
    for ax, rp, duration, func in zip(axs.flat, rps, durations, funcs):
        ax.imshow(rp, cmap="gray", origin="lower")
        ax.set_title(f"{func.__name__}:\n{duration:.6f}")
    time_obj.new("Plotting")
    plt.show()


def view_cdist_vs_time(max_samples: int = 6_000, m: int = 5):
    my_signal = composite_signal(max_samples, ((0.01, 4), (0.02, 2), (0.04, 1)))    # ((1, 4), (2, 2), (4, 1))
    rps = []
    sizes = np.arange(4_000, max_samples+1, 100)
    time_obj = TimeObject()
    for size in sizes:
        for func in (stride_cdist, view_cdist):
            rps.append(func(my_signal[:size], m))
            time_obj.new("")

    # my_len = int(len(time_obj.time_list)/2)

    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f"{view_cdist.__name__}")
    ax.plot(sizes, time_obj.time_list[1::2], label=f"{view_cdist.__name__}")
    ax.plot(sizes, time_obj.time_list[0::2], label=f"{stride_cdist.__name__}")
    ax.set_xlabel("Amount of samples (#)")
    ax.set_ylabel("Time (s)")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # compare_all()
    view_cdist_vs_time()


"""

plt.plot(xs, total_signal)
plt.title("Sum of 3 sine-waves, with frequencies 1, 2 and 4")
plt.show()

10.000 points: 
Setup duration: 5.040192
Convolution duration: 5.585044
Reshape duration: 2.404099
Add 2 halves duration: 2.219247

"""



