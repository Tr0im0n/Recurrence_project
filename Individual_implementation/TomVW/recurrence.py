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

    def new(self, message: str = None):
        new_time = time.time()
        duration = new_time - self.last_time
        print(f"{message} duration: {duration:.6f}")
        self.time_list.append(duration)
        self.last_time = new_time

    def total(self):
        print(f"Total duration: {sum(self.time_list):.6f}")


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


def convolve_triangle_shift(signal: np.ndarray, m: int = 5):
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
    # plt.imshow(half, cmap="gray", origin="lower")
    # plt.show()
    ans = half + half.T
    time_obj.new("Add 2 halves")
    time_obj.total()
    plt.imshow(ans, cmap="gray", origin="lower")
    plt.show()


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
    time_obj.total()
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


if __name__ == "__main__":
    main()
    test2(create_signal1())


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



