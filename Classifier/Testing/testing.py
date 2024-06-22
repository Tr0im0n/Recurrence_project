import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.integrate import solve_ivp
from itertools import groupby
import matplotlib.pyplot as plt
import simdat as sd
import pandas as pd

# -----------------------------------------------------------------------------------------------------------------------------------
# TO-DO:
# 4. Try different event types (look for literature)
# 5. Try bearing dataset
# ------------------------ CLASSIFIER LAYOUT ----------------------------------------------------------------------------------------
# 1. Anomaly Detection
#   - Single rbf-kernel SVM (healthy vs faulty)
# 2. Anomaly Classification
#   - multiple linear-kernel SVMs (comparing each anomaly class to each other one by one)
#   - each SVM classifier votes on one anomaly class => majority anomaly class is chosen
# -----------------------------------------------------------------------------------------------------------------------------------	

def plot_rqa_measures(recurrence_plots, rqa_measures):
    plt.figure(figsize=(12, 18))
    measure_names = list(rqa_measures[0].keys())  # Get measure names from the dictionary keys
    for index, name in enumerate(measure_names):
        measure_data = [measure[name] for measure in rqa_measures]
        ax = plt.subplot(len(measure_names), 1, index + 1)
        ax.plot(measure_data, linestyle='-', color='b')
        ax.set_title(name)
        ax.set_xlabel('Recurrence Plot Index')
        ax.set_ylabel(name)
        ax.grid(True)
    plt.tight_layout()
    plt.show()

def calc_recurrence_plots(timeseries, m, T, epsilon):
    num_vectors = len(timeseries) - (m - 1) * T
    vectors = np.array([timeseries[i:i + m*T:T] for i in range(num_vectors)])
    distance_matrix = squareform(pdist(vectors, metric='euclidean'))
    # Normalize the distance matrix
    max_distance = np.max(distance_matrix)
    if max_distance > 0:
        normalized_distance_matrix = distance_matrix / max_distance
    else:
        normalized_distance_matrix = distance_matrix

    recurrence_matrix = (normalized_distance_matrix <= epsilon).astype(int)
    return recurrence_matrix

def calc_rqa_measures(recurrence_matrix, min_line_length=2):
    time_series_length = recurrence_matrix.shape[0]
    # Calculate recurrence rate (RR)
    RR = np.sum(recurrence_matrix) / (time_series_length ** 2)

    # Calculate diagonal line structures
    diagonals = [np.diag(recurrence_matrix, k) for k in range(-time_series_length + 1, time_series_length)]
    diag_lengths = [len(list(group)) for diag in diagonals for k, group in groupby(diag) if k == 1]

    # Calculate DET
    DET = sum(l for l in diag_lengths if l >= min_line_length) / np.sum(recurrence_matrix) if np.sum(recurrence_matrix) != 0 else 0

    # Calculate L
    L = np.mean([l for l in diag_lengths if l >= min_line_length]) if diag_lengths else 0

    # Calculate Lmax
    Lmax = max(diag_lengths) if diag_lengths else 0

    # Calculate DIV
    DIV = 1 / Lmax if Lmax != 0 else 0

    # Calculate ENTR
    counts = np.bincount(diag_lengths)
    probs = counts / np.sum(counts) if np.sum(counts) > 0 else np.array([0])
    ENTR = -np.sum(probs * np.log(probs + np.finfo(float).eps)) if np.sum(counts) > 0 else 0

    # Calculate trend (TREND)
    TREND = np.mean([np.mean(recurrence_matrix[i, i:]) for i in range(len(recurrence_matrix))])

    # Calculate laminarity (LAM)
    verticals = [recurrence_matrix[:, i] for i in range(time_series_length)]
    vert_lengths = [len(list(group)) for vert in verticals for k, group in groupby(vert) if k == 1]
    LAM = sum(l for l in vert_lengths if l >= min_line_length) / np.sum(recurrence_matrix) if np.sum(recurrence_matrix) != 0 else 0

    # Calculate trapping time (TT)
    # TT = np.mean([l for l in vert_lengths if l >= min_line_length]) if vert_lengths else 0

    # Calculate maximum length of vertical structures (Vmax)
    Vmax = max(vert_lengths) if vert_lengths else 0

    # Calculate entropy of vertical structures (VENTR)
    vert_counts = np.bincount(vert_lengths)
    vert_probs = vert_counts / np.sum(vert_counts) if np.sum(vert_counts) > 0 else np.array([0])
    # VENTR = -np.sum(vert_probs * np.log(vert_probs)) if np.sum(vert_counts) > 0 else 0

    # Ratio between DET and RR
    DET_RR = DET / RR if RR > 0 else 0

    return {'RR': RR, 'DET': DET, 'LAM': LAM, 'DET_RR': DET_RR, 'L': L, 'DIV': DIV, 'ENTR': ENTR, 'TREND': TREND}

def classify_recurrence_plots(rqa_measures, threshold_dict):
    classifications = []
    for measures in rqa_measures:
        if any(measures[measure] > threshold for measure, threshold in threshold_dict.items()):
            classifications.append(1)
        else:
            classifications.append(0)
    return classifications

# ts, spike_locations = sd.composite_signal(1000, ((0.1, 2), (0.19, 1)), noise_amplitude=0.8, return_spike_locations=True)

# ----------------------------
# Bearing Data Set
# ts = pd.read_csv('Classifier/data/normal_3hp_1730rpm.csv')['X100_DE_time'][0:25000].values
# ts = pd.read_csv('Classifier/data/InnerRace_0.028.csv')['X059_DE_time'][0:250000].values
# ts = pd.read_csv('Classifier/data/Ball_0.028.csv')['X051_DE_time'][0:250000].values
# ----------------------------

# synth2
ts = pd.read_csv('Classifier/data/synthetic_fault_data.csv')['Signal'][0:10000].values

#print(ts)
plt.plot(ts)
plt.show()

m = 4 # embedding dimension
T = 2 # delay
epsilon = 0.1 # threshold

l = 100  # Window size
delay = 20  # Delay before calculating next RP

recurrence_plots = []
rqa_measures = []

for start in range(0, len(ts) - l + 1, delay):
    window = ts[start:start + l]
    rp = calc_recurrence_plots(window, m, T, epsilon)
    recurrence_plots.append(rp)
    rqa_metrics = calc_rqa_measures(rp)
    rqa_measures.append(rqa_metrics)

# plt.imshow(recurrence_plots[0], cmap='binary', origin='lower')
# plt.show()

plot_rqa_measures(recurrence_plots, rqa_measures)
plt.show()

# # labeling the recurrence plots
# labels = np.zeros(int((len(ts)-l)/delay))
# for i in range(0, len(recurrence_plots)-1):
#     if any(spike_locations[i*5:i*5+200]):
#         labels[i] = 1

# thresholds = {'RR': 0.1, 'DET': 0.6,'LAM': 0.7, 'ENTR': 1.0, 'TT': 3.0}  
# classifications = classify_recurrence_plots(rqa_measures, thresholds)

# print(f"Classifications: {classifications}")
# print(f"Labels: {labels}")



