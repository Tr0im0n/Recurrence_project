import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
import time

# Read the data from the file
file_path = 'SO_data'
data_all = pd.read_csv(file_path, sep='\s+')

# Extract the SOI column
data = data_all['SOI']

plt.plot(data)
plt.show()


def rp_from_timeseries(epsilon, m, T):
    # Construct embedded vectors
    num_vectors = len(data) - (m - 1) * T
    vectors = np.array([data[t:t + m * T:T] for t in range(num_vectors)])

    # Find recurrence points using distance between embedded vectors
    D = squareform(pdist(vectors, metric='euclidean'))
    hit = np.argwhere(D < epsilon)

    # Extract x and y coordinates of points of recurrence
    x_rec, y_rec = hit[:, 0], hit[:, 1]

    return (D, x_rec, y_rec)

def plot_distance_matrix(D):
    # Plot the distance matrix D
    plt.figure(figsize=(10, 8))
    plt.imshow(D, origin='lower', cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Distance Matrix')
    plt.xlabel('Vector Index')
    plt.ylabel('Vector Index')
    plt.show()

def plot_rp(x_rec, y_rec):
    # Plot points of recurrence
    plt.figure(figsize=(10, 8))
    plt.scatter(x_rec, y_rec, s=1)
    plt.title('Recurrence Plot')
    plt.xlabel('Vector Index')
    plt.ylabel('Vector Index')
    plt.show()

# Define parameters
epsilon = 4.406
m = 3            # embedding dimension
T = 1               # time delay

D, x_rec, y_rec = rp_from_timeseries(epsilon, m, T)
plot_distance_matrix(D)
plot_rp(x_rec, y_rec)




