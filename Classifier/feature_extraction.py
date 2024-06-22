
import numpy as np
from scipy.spatial.distance import pdist, squareform
from itertools import groupby

def calc_recurrence_plots(timeseries, m, T, epsilon):
    num_vectors = len(timeseries) - (m - 1) * T
    vectors = np.array([timeseries[i:i + m*T:T] for i in range(num_vectors)])
    distance_matrix = squareform(pdist(vectors, metric='euclidean'))
    max_distance = np.max(distance_matrix)
    normalized_distance_matrix = distance_matrix / max_distance if max_distance > 0 else distance_matrix
    recurrence_matrix = (normalized_distance_matrix <= epsilon).astype(int)
    return recurrence_matrix

def calc_rqa_measures(recurrence_matrix, min_line_length=2):
    time_series_length = recurrence_matrix.shape[0]
    RR = np.sum(recurrence_matrix) / (time_series_length ** 2)
    diagonals = [np.diag(recurrence_matrix, k) for k in range(-time_series_length + 1, time_series_length)]
    diag_lengths = [len(list(group)) for diag in diagonals for k, group in groupby(diag) if k == 1]
    DET = sum(l for l in diag_lengths if l >= min_line_length) / np.sum(recurrence_matrix) if np.sum(recurrence_matrix) != 0 else 0
    L = np.mean([l for l in diag_lengths if l >= min_line_length]) if diag_lengths else 0
    Lmax = max(diag_lengths) if diag_lengths else 0
    DIV = 1 / Lmax if Lmax != 0 else 0
    counts = np.bincount(diag_lengths)
    probs = counts / np.sum(counts) if np.sum(counts) > 0 else np.zeros_like(counts)
    ENTR = -np.sum(p * np.log2(p) for p in probs if p > 0)
    vertical_lengths = []
    for j in range(recurrence_matrix.shape[1]):  # For each column in the matrix
        column = recurrence_matrix[:, j]
        vertical_lengths.extend(get_vertical_line_lengths(column))
    
    TT = np.mean(vertical_lengths) if vertical_lengths else 0  # Calculate the average if not empty
    return {'RR': RR, 'DET': DET, 'L': L, 'TT': TT, 'Lmax': Lmax, 'DIV': DIV, 'ENTR': ENTR}

def get_vertical_line_lengths(column):
    lengths = []
    current_length = 0
    for value in column:
        if value == 1:
            current_length += 1
        elif current_length > 0:
            lengths.append(current_length)
            current_length = 0
    if current_length > 0:
        lengths.append(current_length)
    return lengths
