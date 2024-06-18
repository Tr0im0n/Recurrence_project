import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.integrate import solve_ivp
from itertools import groupby
import matplotlib.pyplot as plt
import simdat as sd

def calcRP(timeseries, m, T, epsilon):
    l = timeseries.shape[0]
    ones = np.ones_like(timeseries)

    H = np.zeros((l-m+1, m)) # Trajectory Matrix
    for i in range(l-m*T+1):
        H[i] = timeseries[i:i+m*T:T]

    P = np.kron(ones, H) - np.kron(H, ones) 
    distance_matrix = squareform(pdist(P, 'euclidean'))

    # Normalize the distance matrix
    max_distance = np.max(distance_matrix)
    if max_distance > 0:
        normalized_distance_matrix = distance_matrix / max_distance
    else:
        normalized_distance_matrix = distance_matrix

    recurrence_matrix = (normalized_distance_matrix <= epsilon).astype(int)
    return recurrence_matrix

def calcRQAMeasures(rp):
    n = rp.shape[0]
    rr = np.sum(rp) / (n * n)  # Recurrence Rate

    # Finding diagonals (at least 2 points)
    diagonals = np.array([np.sum(np.diagonal(rp, offset=i)) for i in range(1, n)])
    det = np.sum(diagonals) / np.sum(rp) if np.sum(rp) != 0 else 0  # Determinism

    # Average Diagonal Line Length
    diag_lengths = np.array([len(list(g)) for offset in range(-n + 1, n) for k, g in groupby(np.diagonal(rp, offset)) if k])
    l = np.mean(diag_lengths) if len(diag_lengths) > 0 else 0

    # Divergence
    div = 1 / np.max(diag_lengths) if len(diag_lengths) > 0 else 0

    return rr, det, l, div  # Recurrence Rate, Determinism, Average Diagonal Line Length, Divergence

timeseries = sd.composite_signal(1000, ((0.1, 2), (0.19, 1)), noise_amplitude=0.8)

m = 4 # embedding dimension
T = 2 # delay
epsilon = 0.1 # threshold

l = 200  # Window size
delay = 200  # Delay before calculating next RP

recurrence_plots = []
rqa_measures = []
for start in range(0, len(timeseries) - l + 1, delay):
    window = timeseries[start:start + l]
    rp = calcRP(window, m, T, epsilon)
    recurrence_plots.append(rp)
    rqa_metrics = calcRQAMeasures(rp)
    rqa_measures.append(rqa_metrics)

plt.imshow(recurrence_plots[1], cmap='binary', origin='lower')
plt.title('Recurrence Plot')
plt.xlabel('Time Steps')
plt.ylabel('Time Steps')
plt.show()