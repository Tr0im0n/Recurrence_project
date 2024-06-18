import time

import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib.widgets import Slider, Button
from scipy.spatial.distance import pdist, squareform
from scipy.signal import convolve2d

from Individual_implementation.TomVW.synthetic import composite_signal


class TimeObject:
    def __init__(self):
        self.last_time = time.time()
        self.time_list = []

    def new(self, message: str = None):
        new_time = time.time()
        duration = new_time - self.last_time
        print(f"{message} duration: {duration:.6f}")
        self.time_list.append(duration)
        self.last_time = new_time

    def total(self):
        print(f"Total duration: {sum(self.time_list):.6f}")


def get_length_matrix(signal: np.ndarray):
    """ Just used in following function """
    length = signal.shape[0]
    ans = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            my_len = pow(signal[i] - signal[j], 2)
            ans[i, j] = my_len
    return ans


def rp_double_for_loop(signal: np.ndarray, m: int = 5):
    """ Double for loop implementation """
    length = signal.shape[0]
    signal = signal.reshape(length, 1)
    length_matrix = squareform(pdist(signal))
    ans = np.zeros_like(length_matrix)

    for i in range(length-m):
        for j in range(length-m):
            my_len = sum([length_matrix[i+k, j+k] for k in range(m)])
            # clip = 0 if my_len < 0.1 else 1
            ans[i, j] = my_len  # clip
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
    flattened = my_zeros.flatten()
    upper_left_triangle = flattened[:-1].reshape((my_length-1, my_length+1))
    kernel = np.ones((m, 1))
    convolved = convolve2d(upper_left_triangle, kernel, "valid")
    flattened = convolved.flatten()
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
    flattened = my_zeros.flatten()
    upper_left_triangle = flattened[:-1].reshape((my_length-1, my_length+1))
    kernel = np.ones((m, 1))
    time_obj.new("Reshape Triangle")
    convolved = convolve2d(upper_left_triangle, kernel, "valid")
    time_obj.new("Convolution")
    flattened = convolved.flatten()
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
    time_obj = TimeObject()
    distances = pdist(signal, metric='euclidean')   # [:, np.newaxis]
    time_obj.new("pdist")
    distance_matrix = squareform(distances)
    time_obj.new("squareform")
    # distance_matrix = squareform(pdist(signal, "euclidean"))
    kernel = np.eye(m, dtype=int)
    convolved = convolve2d(distance_matrix, kernel)
    time_obj.new("convolve")
    time_obj.total()
    return convolved


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


def compare4():
    my_signal = composite_signal(1_000, ((0.01, 4), (0.02, 2), (0.04, 1)))    # ((1, 4), (2, 2), (4, 1))
    funcs = [hankel_pdist, convolve_diagonal]     # , hankel_kron_norm, rp_double_for_loop, convolve_triangle_shift
    # rps = [func(my_signal) for func in funcs]
    rps = []
    time_obj = TimeObject()
    for func in funcs:
        rps.append(func(my_signal))
        time_obj.new("")

    fig1, axs = plt.subplots(1, 2)
    for ax, rp in zip(axs.flat, rps):
        ax.imshow(rp, cmap="gray", origin="lower")
    plt.show()


if __name__ == "__main__":
    compare4()


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



