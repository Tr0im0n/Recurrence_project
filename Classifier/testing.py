import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.integrate import solve_ivp
from itertools import groupby
import matplotlib.pyplot as plt
import simdat as sd

def plot_rqa_measures(recurrence_plots, rqa_measures):
    # Extracting each RQA measure
    rrs = [measure[0] for measure in rqa_measures]  # Recurrence Rates
    dets = [measure[1] for measure in rqa_measures]  # Determinism
    ls = [measure[2] for measure in rqa_measures]  # Average Diagonal Line Length
    divs = [measure[3] for measure in rqa_measures]  # Divergence

    # Create separate plots for each RQA measure
    measures = [rrs, dets, ls, divs]
    titles = ['Recurrence Rate (RR)', 'Determinism (DET)', 'Average Diagonal Line Length (L)', 'Divergence (DIV)']
    y_labels = ['RR', 'DET', 'L', 'DIV']

    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    for i, (measure, title, label) in enumerate(zip(measures, titles, y_labels)):
        ax = plt.subplot(4, 1, i + 1)
        ax.plot(measure, marker='o', linestyle='-', color='b')
        ax.set_title(title)
        ax.set_xlabel('Recurrence Plot Index')
        ax.set_ylabel(label)
        ax.grid(True)

    plt.tight_layout()
    plt.show()

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

m = 3 # embedding dimension
T = 2 # delay
epsilon = 0.1 # threshold

l = 200  # Window size
delay = 5  # Delay before calculating next RP

recurrence_plots = []
rqa_measures = []
for start in range(0, len(timeseries) - l + 1, delay):
    window = timeseries[start:start + l]
    rp = calcRP(window, m, T, epsilon)
    recurrence_plots.append(rp)
    rqa_metrics = calcRQAMeasures(rp)
    rqa_measures.append(rqa_metrics)

plt.plot(timeseries)
plt.show()

plot_rqa_measures(recurrence_plots, rqa_measures)