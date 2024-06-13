import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy
from pyunicorn.timeseries import RecurrenceNetwork, RecurrencePlot

# Read the data from the file
file_path = 'sensor_data.csv'
data_all = pd.read_csv(file_path)

# Extract the column
variable = 'vibration'
data = data_all[variable].values

data_trunc = data[:1000]


def rp_from_timeseries(data, epsilon, m, T):
    # Construct embedded vectors
    num_vectors = len(data) - (m - 1) * T
    vectors = np.array([data[t:t + m * T:T] for t in range(num_vectors)])

    # Find recurrence points using distance between embedded vectors
    D = squareform(pdist(vectors, metric='euclidean'))
    hit = np.argwhere(D < epsilon)

    # Extract x and y coordinates of points of recurrence
    x_rec, y_rec = hit[:, 0], hit[:, 1]

    return (D, x_rec, y_rec)

def rp_fast(data, epsilon, m=5):
    length = len(data)
    hankel = scipy.linalg.hankel(data[:length - m + 1], data[length - m:])
    ones_like = np.ones_like(data)
    diff_vect = np.kron(ones_like, hankel) - np.kron(hankel, ones_like)
    recurrence_matrix = squareform(pdist(diff_vect, metric='euclidean'))
    recurrence_thresh = (recurrence_matrix <= epsilon).astype(int)

    return recurrence_thresh


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
    plt.title(f'Recurrence Plot of {variable}')
    plt.xlabel('Vector Index')
    plt.ylabel('Vector Index')
    plt.show()

# Define parameters
epsilon = 60
m = 10            # embedding dimension
T = 5               # time delay


recurrence_thresh = rp_fast(data_trunc, m, epsilon)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(data_trunc)
ax1.set_title("My signal")
ax2.imshow(recurrence_thresh, cmap="gray", origin="lower")   # [:990, :990]
ax2.set_title("Clipped recurrence plot")
plt.show()

# D, x_rec, y_rec = rp_from_timeseries(data_trunc, epsilon, m, T)
# plot_distance_matrix(D)
# plot_rp(x_rec, y_rec)