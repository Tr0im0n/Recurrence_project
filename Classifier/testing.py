import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.integrate import solve_ivp
from itertools import groupby
import matplotlib.pyplot as plt
import simdat as sd

# --------------------------------------------------------
# TO-DO:
# 1. verify correctness of rqa measures
# 2. Check each RQA-M reaction to spike type event
# 3. implement decision tree like classifier (If RR > 0.5 then 1 else 0 etc.)
# 4. Try different event types
# 5. Set up WOA-SVM training and testing
# --------------------------------------------------------

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

def calcRQAMeasures(rp, min_line_length=2):
    n = rp.shape[0]
    # Compute Recurrence Rate (RR)
    RR = np.sum(rp) / (n * n)

    # Finding diagonals for Determinism and Average Diagonal Line Length
    diagonals = []
    for offset in range(-n + 1, n):
        diagonal = np.diagonal(rp, offset=offset)
        for k, g in groupby(diagonal):
            if k == 1:
                length = len(list(g))
                if length >= min_line_length:
                    diagonals.append(length)
                    
    if diagonals:
        DET = sum(diagonals) / np.sum(rp)  # Determinism
        L = np.mean(diagonals)             # Average Diagonal Line Length
        DIV = 1 / max(diagonals)           # Divergence
    else:
        DET = 0
        L = 0
        DIV = 0

    # Compute Laminarity and Trapping Time from vertical lines
    verticals = []
    for j in range(n):
        column = rp[:, j]
        for k, g in groupby(column):
            if k == 1:
                length = len(list(g))
                if length >= min_line_length:
                    verticals.append(length)
    
    if verticals:
        TT = np.mean(verticals)  # Trapping Time
        LAM = sum(verticals) / np.sum(rp)  # Laminarity
    else:
        TT = 0
        LAM = 0

    # Calculate Entropy of diagonal line lengths
    if diagonals:
        line_length_counts = np.bincount(diagonals)
        p = line_length_counts[line_length_counts.nonzero()] / sum(line_length_counts)
        ENT = -np.sum(p * np.log(p))
    else:
        ENT = 0

    # Calculate Trend
    trend_lines = [np.sum(np.diagonal(rp, offset=i)) for i in range(-n + 1, n)]
    trend_slope, _ = np.polyfit(range(len(trend_lines)), trend_lines, 1)
    TREND = trend_slope

    # Ratio between DET and RR
    DET_RR = DET / RR if RR > 0 else 0

    return RR, DET, LAM, DET_RR, L, TT, DIV, ENT, TREND

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