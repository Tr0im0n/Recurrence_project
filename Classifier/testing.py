import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.integrate import solve_ivp
from itertools import groupby
import matplotlib.pyplot as plt
import simdat as sd

# -----------------------------------------------------------------------------------------------------------------------------------
# TO-DO:
# 1. verify correctness of rqa measures (Fix ENTR)
# 3. implement decision tree like classifier (If RR > 0.5 then 1 else 0 etc.)
# 4. Try different event types (look for literature)
# 5. Try bearing dataset
# 6. label recurrence plots as faulty/healthy based on the array containing the spike coordinates.
# ------------------------ CLASSIFIER LAYOUT ----------------------------------------------------------------------------------------
# 1. Anomaly Detection
#   - Single rbf-kernel SVM (healthy vs faulty)
# 2. Anomaly Classification
#   - multiple linear-kernel SVMs (comparing each anomaly class to each other one by one)
#   - each SVM classifier votes on one anomaly class => majority anomaly class is chosen
# -----------------------------------------------------------------------------------------------------------------------------------	

def plot_rqa_measures(recurrence_plots, rqa_measures):
    # Create separate plots for each RQA measure
    measure_names = ['Recurrence Rate', 'Determinism', 'Laminarity', 'Ratio DET/RR', 'Avg Diagonal Length', 'Trapping Time', 'Divergence', 'Entropy', 'Trend']
    plt.figure(figsize=(12, 18))  # Adjust the figure size as needed
    for index, name in enumerate(measure_names):
        measure_data = [measure[index] for measure in rqa_measures]
        ax = plt.subplot(len(measure_names), 1, index + 1)
        ax.plot(measure_data, linestyle='-', color='b')
        ax.set_title(name)
        ax.set_xlabel('Recurrence Plot Index')
        ax.set_ylabel(name)
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

def calcRQAMeasures(recurrence_matrix, min_line_length=2):
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
    # ENTR = -np.sum(probs * np.log(probs)) if np.sum(counts) > 0 else 0

    # Calculate trend (TREND)
    TREND = np.mean([np.mean(recurrence_matrix[i, i:]) for i in range(len(recurrence_matrix))])

    # Calculate laminarity (LAM)
    verticals = [recurrence_matrix[:, i] for i in range(time_series_length)]
    vert_lengths = [len(list(group)) for vert in verticals for k, group in groupby(vert) if k == 1]
    LAM = sum(l for l in vert_lengths if l >= min_line_length) / np.sum(recurrence_matrix) if np.sum(recurrence_matrix) != 0 else 0

    # Calculate trapping time (TT)
    TT = np.mean([l for l in vert_lengths if l >= min_line_length]) if vert_lengths else 0

    # Calculate maximum length of vertical structures (Vmax)
    Vmax = max(vert_lengths) if vert_lengths else 0

    # Calculate entropy of vertical structures (VENTR)
    vert_counts = np.bincount(vert_lengths)
    vert_probs = vert_counts / np.sum(vert_counts) if np.sum(vert_counts) > 0 else np.array([0])
    # VENTR = -np.sum(vert_probs * np.log(vert_probs)) if np.sum(vert_counts) > 0 else 0

    # Ratio between DET and RR
    DET_RR = DET / RR if RR > 0 else 0

    return RR, DET, LAM, DET_RR, L, TT, DIV, TREND

def detect_changes(rqa_measures, threshold=0.2, window_size=5):
    detected_points = {
        'RR': [],
        'DET': [],
        'LAM': []
    }
    for i in range(window_size, len(rqa_measures)):
        for measure, name in zip([0, 1, 2], ['RR', 'DET', 'LAM']):
            current_value = rqa_measures[i][measure]
            avg_past_values = np.mean([rqa_measures[j][measure] for j in range(i - window_size, i)])
            if abs(current_value - avg_past_values) > threshold * avg_past_values:
                detected_points[name].append(i)  # Record the index of the RQA measure change
    return detected_points

timeseries, spike_locations = sd.composite_signal(1000, ((0.1, 2), (0.19, 1)), noise_amplitude=0.8, return_spike_locations=True)

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

# labeling the recurrence plots
labels = np.zeros(int((len(timeseries)-l)/delay))
for i in range(0, len(recurrence_plots)-1):
    if any(spike_locations[i*5:i*5+200]):
        labels[i] = 1

print(labels)

plt.plot(timeseries)
plt.show()
# plot_rqa_measures(recurrence_plots, rqa_measures)

# Detect changes in RR, DET, LAM
# detected_points = detect_changes(rqa_measures)
# print("Detected points for RR:", detected_points['RR'])
# print("Detected points for DET:", detected_points['DET'])
# print("Detected points for LAM:", detected_points['LAM'])
