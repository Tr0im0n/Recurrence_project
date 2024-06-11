# Martina Kiewek
# June, 2024
# Creating Recurrence plots and performing basic RQA

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
import time


# Create sine wave time series
length = 20
x = np.arange(-length, length, 0.1)
y = np.sin(x)

plt.plot(y)
plt.show()

# y = [1,2,3,4,5,6,7,8,9,10]

def rp_from_timeseries(epsilon, m, T):
    # Construct embedded vectors
    num_vectors = len(y) - (m - 1) * T
    vectors = np.array([y[t:t + m * T:T] for t in range(num_vectors)])

    # Find recurrence points using distance between embedded vectors
    # start_time = time.perf_counter()

    # D = np.zeros((num_vectors, num_vectors))  # Initialize similarity matrix
    # for i in range(num_vectors):
    #     for j in range(num_vectors):
    #         diff_vector = vectors[i] - vectors[j]
    #         len_difference = np.linalg.norm(diff_vector)
    #         D[i, j] = len_difference

    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # print('method 1: ', elapsed_time)

    # start_time = time.perf_counter()
    D = squareform(pdist(vectors, metric='euclidean'))
    end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # print('method 2: ', elapsed_time)

    # Trial showed method 1 ~ 0.937 s while method 2 ~ 0.00292 s

    # Find points of recurrence
    hit = np.argwhere(D < epsilon)

    # Extract x and y coordinates of points of recurrence
    x_rec, y_rec = hit[:, 0], hit[:, 1]

    # Plot the distance matrix D
    plt.figure(figsize=(10, 8))
    plt.imshow(D, origin='lower', cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Distance Matrix')
    plt.xlabel('Vector Index')
    plt.ylabel('Vector Index')
    # plt.show()

    # Plot points of recurrence
    plt.figure(figsize=(10, 8))
    plt.scatter(x_rec, y_rec, s=1)
    plt.title('Recurrence Plot')
    plt.xlabel('Vector Index')
    plt.ylabel('Vector Index')
    plt.show()

    return(D, x_rec, y_rec)

# percent of total points in plot at which recurrence was found
def recurrence_rate(D):
    num_rec = len(x_rec)
    num_total = (np.shape(D)[0])**2
    RR = num_rec/num_total
    return(RR)

def determinism(D, x_rec, y_rec):
    coord_pairs = [(x_rec[i], y_rec[i]) for i in range(len(x_rec))]
    lengths = []
    visited = set()

    for pair in (coord_pairs):
        if pair in visited:
            continue

        length = 0
        x, y = pair

        start_x, start_y = x, y

        while ((x + 1), (y + 1)) in coord_pairs:
            length += 1
            x += 1
            y += 1
            visited.add((x, y))

        if length > 0:  # Only append lengths greater than 0
            lengths.append(length)
        visited.add((start_x, start_y))

    print(lengths)

    # plt.boxplot(lengths)
    # plt.title("Lengths of Diagonal Lines in Recurrence Plot")
    # plt.xlabel("Line Lengths")
    # plt.ylabel("Frequency")
    # plt.show()


# Define parameters
epsilon = .11
m = 6            # embedding dimension
T = 3               # time delay

D, x_rec, y_rec = rp_from_timeseries(epsilon, m, T)
# recurrence_rate(D)
determinism(D, x_rec, y_rec)
