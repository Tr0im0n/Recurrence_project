
import numpy as np
from scipy.spatial.distance import pdist, squareform
from itertools import groupby
from sklearn.neighbors import NearestNeighbors

def calc_recurrence_plots(timeseries, m, T, epsilon, use_fnn=False):
    if use_fnn:
        m = false_nearest_neighbors(timeseries)
        print(f"Estimated embedding dimension m using FNN: {m}")

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
    # ENTR
    counts = np.bincount(diag_lengths)
    probs = counts / np.sum(counts) if np.sum(counts) > 0 else np.zeros_like(counts)
    ENTR = -np.sum(p * np.log2(p) for p in probs if p > 0)
    # TT
    vertical_lengths = []
    for j in range(recurrence_matrix.shape[1]):  # For each column in the matrix
        column = recurrence_matrix[:, j]
        vertical_lengths.extend(get_vertical_line_lengths(column))
    TT = np.mean(vertical_lengths) if vertical_lengths else 0  # Calculate the average if not empty
    # Calculate laminarity (LAM)
    verticals = [recurrence_matrix[:, i] for i in range(time_series_length)]
    vert_lengths = [len(list(group)) for vert in verticals for k, group in groupby(vert) if k == 1]
    LAM = sum(l for l in vert_lengths if l >= min_line_length) / np.sum(recurrence_matrix) if np.sum(recurrence_matrix) != 0 else 0
    return {'RR': RR, 'DET': DET, 'L': L, 'TT': TT, 'Lmax': Lmax, 'DIV': DIV, 'ENTR': ENTR, 'LAM': LAM}

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

def false_nearest_neighbors(timeseries, max_dim: int = 15, T=1, Rtol=10.0, Atol=2.0):
    n = len(timeseries)

    for d in range(1, max_dim + 1):
        embedding = np.zeros((n - d * T + 1, d))
        for i in range(n - d * T + 1):
            embedding[i] = timeseries[i:i + d * T:T]

        if embedding.shape[0] < 2:
            continue  # Need at least two points to proceed

        # Find nearest neighbors in the current dimension
        nn = NearestNeighbors(n_neighbors=2).fit(embedding)
        distances, indices = nn.kneighbors(embedding)

        count_fnn = 0
        for i in range(len(embedding)):
            dist_d = distances[i, 1]
            neighbor_idx = indices[i, 1]

            # Check next higher dimension
            if i + d * T < n and neighbor_idx + d * T < n:
                point_d1 = np.append(embedding[i], timeseries[i + d * T])
                neighbor_point_d1 = np.append(embedding[neighbor_idx], timeseries[neighbor_idx + d * T])
                dist_d1 = np.linalg.norm(point_d1 - neighbor_point_d1)

                if dist_d1 / dist_d > Rtol or abs(dist_d1 - dist_d) > Atol:
                    count_fnn += 1

        fnn_ratio = count_fnn / len(embedding)
        if fnn_ratio < 0.1:
            return d  # Return the current dimension as the minimum

    return -1  # Return -1 if no suitable dimension is found