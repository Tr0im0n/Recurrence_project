
import numpy as np
from scipy.spatial.distance import pdist, squareform
from itertools import groupby
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy

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
    
    # Recurrence Rate (RR)
    RR = np.mean(recurrence_matrix)
    
    # Extract diagonal lines
    diags = [np.diag(recurrence_matrix, k) for k in range(-time_series_length + 1, time_series_length)]
    
    # Calculate lengths of diagonal lines
    diag_lengths = np.concatenate([np.diff(np.where(np.concatenate(([d[0]],d,[-1])))[0])[::2] for d in diags])
    
    # Filter diagonal lines by minimum length
    valid_diag_lengths = diag_lengths[diag_lengths >= min_line_length]
    
    # Determinism (DET)
    DET = np.sum(valid_diag_lengths) / np.sum(recurrence_matrix) if np.sum(recurrence_matrix) != 0 else 0
    
    # Average diagonal line length (L)
    L = np.mean(valid_diag_lengths) if len(valid_diag_lengths) > 0 else 0
    
    # Longest diagonal line (Lmax)
    Lmax = np.max(diag_lengths) if len(diag_lengths) > 0 else 0
    
    # Divergence (DIV)
    DIV = 1 / Lmax if Lmax != 0 else 0
    
    # Entropy of diagonal line lengths (ENTR)
    if len(valid_diag_lengths) > 0:
        unique, counts = np.unique(valid_diag_lengths, return_counts=True)
        ENTR = entropy(counts)
    else:
        ENTR = 0
    
    # Laminarity (LAM) and Trapping Time (TT)
    vertical_lengths = []
    for col in range(time_series_length):
        v_lines = np.diff(np.where(np.concatenate(([recurrence_matrix[0, col]],
                                                   recurrence_matrix[:, col],
                                                   [not recurrence_matrix[-1, col]])))[0])[::2]
        vertical_lengths.extend(v_lines)
    
    vertical_lengths = np.array(vertical_lengths)
    valid_vertical_lengths = vertical_lengths[vertical_lengths >= min_line_length]
    
    LAM = np.sum(valid_vertical_lengths) / np.sum(recurrence_matrix) if np.sum(recurrence_matrix) != 0 else 0
    TT = np.mean(valid_vertical_lengths) if len(valid_vertical_lengths) > 0 else 0
    return np.array([RR, DET, L, TT, Lmax, DIV, ENTR, LAM])

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


def false_nearest_neighbors2(series, max_dim, tau):
    num_points = len(series)

    # Initialize arrays to store results
    fnn_ratios = np.zeros(max_dim - 1)

    for m in range(1, max_dim):
        # Time delay embedding
        X_embedded = np.asarray([series[i:i + m] for i in range(num_points - (m - 1) * tau)])

        # Calculate distances in embedded space
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X_embedded)
        distances, _ = nbrs.kneighbors(X_embedded)

        # Calculate Euclidean distances
        dist1 = distances[:, 1]  # distance to the nearest neighbor in embedded space
        dist_original = np.abs(series[:len(dist1)] - series[m * tau:])  # distance in original space

        # Compute the ratio of distances
        ratio = dist1 / dist_original

        # Count false nearest neighbors
        fnn_count = np.sum(ratio > 15)  # threshold for false nearest neighbors

        # Calculate ratio of false nearest neighbors
        fnn_ratio = fnn_count / len(series)

        # Store the ratio for this embedding dimension
        fnn_ratios[m - 1] = fnn_ratio

    return fnn_ratios


def test1():
    # Example usage
    # Generate a sample time series
    np.random.seed(0)
    time_series = np.cumsum(np.random.randn(1000))

    # Parameters
    max_dim = 10  # maximum embedding dimension to check
    tau = 1  # time delay

    # Compute false nearest neighbor ratios
    fnn_ratios = false_nearest_neighbors(time_series, max_dim, tau)

    # Print the results
    print("False nearest neighbor ratios:")
    for m in range(1, max_dim):
        print(f"Dimension {m}: {fnn_ratios[m - 1]}")
