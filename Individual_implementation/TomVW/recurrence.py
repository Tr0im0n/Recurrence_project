import time

import numpy as np
import matplotlib.pyplot as plt


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


def create_signal1(show: bool = False):
    xs = np.arange(0, 10, 0.01)
    signal1 = 4*np.sin(np.pi*xs)
    signal2 = 2*np.sin(2*np.pi*xs)
    signal3 = np.sin(4*np.pi*xs)
    total_signal = signal1 + signal2 + signal3
    if show:
        plt.plot(xs, total_signal)
        plt.title("Sum of 3 sine-waves, with frequencies 1, 2 and 4")
        plt.show()
    return total_signal


def get_length_matrix(signal: np.ndarray, show: bool = False):
    length = signal.shape[0]
    ans = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            my_len = pow(signal[i] - signal[j], 2)
            ans[i, j] = my_len
    if show:
        plt.imshow(ans)
        plt.show()
    return ans


def recurrence_matrix(length_matrix: np.ndarray):
    length = length_matrix.shape[0]
    ans = np.zeros_like(length_matrix)

    samples = 10
    for i in range(length-samples):
        for j in range(length-samples):
            my_len = sum([length_matrix[i+k, j+k] for k in range(samples)])
            clip = 0 if my_len < 0.1 else 1
            ans[i, j] = clip
    return ans


def main():
    time_obj = TimeObject()
    my_signal = create_signal1(False)
    time_obj.new()
    my_length_matrix = get_length_matrix(my_signal, True)
    time_obj.new()
    my_recurrence_matrix = recurrence_matrix(my_length_matrix)
    time_obj.new()
    plt.imshow(my_recurrence_matrix[:990, :990], cmap="gray", origin="lower")
    plt.title("Clipped recurrence plot")
    plt.show()


if __name__ == "__main__":
    main()
