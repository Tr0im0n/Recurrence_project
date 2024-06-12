import time

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.spatial.distance import pdist, squareform


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
    xs = np.arange(0, 10, 0.1)
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
    my_hankel = scipy.linalg.hankel(signal[:my_length - n], signal[my_length - n:])
    my_ones_like = np.ones_like(signal)
    whatsthis = np.kron(my_ones_like, my_hankel) - np.kron(my_hankel, my_ones_like)
    return squareform(pdist(whatsthis, metric='euclidean'))


def main():
    time_obj = TimeObject()
    my_signal = create_signal1()
    time_obj.new()
    # my_length_matrix = get_length_matrix(my_signal)
    # time_obj.new()
    # my_recurrence_matrix = recurrence_matrix_slow(my_length_matrix)
    my_recurrence_matrix = fast(my_signal)
    time_obj.new()
    epsilon = 0.2
    my_r = (my_recurrence_matrix <= epsilon).astype(int)
    time_obj.new()
    plt.imshow(my_r, cmap="binary", origin="lower")   # [:990, :990]
    plt.title("Clipped recurrence plot")
    plt.show()


if __name__ == "__main__":
    main()


"""

plt.plot(xs, total_signal)
plt.title("Sum of 3 sine-waves, with frequencies 1, 2 and 4")
plt.show()

"""



