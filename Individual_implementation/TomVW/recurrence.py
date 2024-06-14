import time

import numpy as np
import matplotlib.pyplot as plt
import scipy
from matplotlib.widgets import Slider, Button
from scipy.spatial.distance import pdist, squareform
from scipy.signal import convolve2d


class TimeObject:
    def __init__(self):
        self.last_time = time.time()
        self.time_list = []

    def new(self):
        new_time = time.time()
        duration = new_time - self.last_time
        print(f"Duration: {duration}")
        self.time_list.append(duration)
        self.last_time = new_time


def create_signal1():
    xs = np.arange(0, 10, 0.01)
    signal1 = 4*np.sin(np.pi*xs)
    signal2 = 2*np.sin(2*np.pi*xs)
    signal3 = np.sin(4*np.pi*xs)
    total_signal = signal1 + signal2 + signal3
    return total_signal


def get_length_matrix(signal: np.ndarray):
    """ Just used in following function """
    length = signal.shape[0]
    ans = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            my_len = pow(signal[i] - signal[j], 2)
            ans[i, j] = my_len
    return ans


def recurrence_matrix_slow(length_matrix: np.ndarray):
    """ Double for loop implementation """
    length = length_matrix.shape[0]
    ans = np.zeros_like(length_matrix)

    samples = 10
    for i in range(length-samples):
        for j in range(length-samples):
            my_len = sum([length_matrix[i+k, j+k] for k in range(samples)])
            clip = 0 if my_len < 0.1 else 1
            ans[i, j] = clip
    return ans


def fast(signal: np.ndarray, n: int = 5):
    """
    :param signal:
    :param n: amount of points we look at
    :return:
    """
    my_length = signal.shape[0]
    my_hankel = scipy.linalg.hankel(signal[:my_length - n + 1], signal[my_length - n:])
    my_ones = np.ones((my_length - n + 1))
    whatsthis = np.kron(my_ones, my_hankel) - np.kron(my_hankel, my_ones)
    return squareform(pdist(whatsthis, metric='euclidean'))


def test2(signal: np.ndarray, m: int = 5):
    """
    Trying to implement my own recurrence plot.
    :param signal: np array of the signal
    :param m: The embedding dimension
    :return: The Recurrence plot, clipped?
    """
    my_length = signal.shape[0]
    signal.reshape(my_length, 1)
    my_distances = pdist(signal, "euclidean")
    my_zeros = np.zero((my_length, my_length - 1))
    rows, cols = np.triu_indices(my_zeros, k=0)
    my_zeros[rows, cols] = my_distances
    my_zeros.reshape((my_length - 1, my_length))


def convolve_diagonal(signal: np.ndarray, m: int = 5):
    """
    Trying to implement my own recurrence plot.
    :param signal: np array of the signal
    :param m: The embedding dimension
    :return: The Recurrence plot, clipped?
    """
    my_length = signal.shape[0]
    signal.reshape(my_length, 1)
    time_obj = TimeObject()
    distances = pdist(signal[:, np.newaxis], metric='euclidean')
    time_obj.new()
    distance_matrix = squareform(distances)
    time_obj.new()
    # distance_matrix = squareform(pdist(signal, "euclidean"))
    kernel = np.eye(m, dtype=int)
    convolved = convolve2d(distance_matrix, kernel)
    time_obj.new()
    plt.imshow(convolved, cmap="gray", origin="lower")
    plt.show()


def main():
    time_obj = TimeObject()
    my_signal = create_signal1()
    time_obj.new()
    # my_length_matrix = get_length_matrix(my_signal)
    # time_obj.new()
    # my_recurrence_matrix = recurrence_matrix_slow(my_length_matrix)
    my_recurrence_matrix = fast(my_signal)
    time_obj.new()
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


if __name__ == "__main__":
    # main()
    convolve_diagonal(create_signal1())


"""

plt.plot(xs, total_signal)
plt.title("Sum of 3 sine-waves, with frequencies 1, 2 and 4")
plt.show()

"""



