
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from itertools import groupby
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.neighbourhood import FixedRadius
from pyrqa.computation import RQAComputation
from pyrqa.metric import EuclideanMetric

def calc_recurrence_plot(timeseries: np.ndarray, m: int, T: int, epsilon: float = 0.1, use_fnn: bool = False):
    if use_fnn:
        m = false_nearest_neighbors(timeseries, tau = T)
        print(f"Estimated embedding dimension m using FNN: {m}")

    new_shape = timeseries.shape[0] - (m - 1) * T
    indices = np.arange(new_shape)[:, None] + np.arange(0, m * T, T)    # new_shape x m
    result = timeseries[indices]    # just a view
    distance_matrix = cdist(result, result, metric='euclidean')    # new_shape x new_shape

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

def pyrqa(timeseries, m, T, epsilon):
    time_series = TimeSeries(timeseries,
                                   embedding_dimension=m,
                                   time_delay=T)
    settings = Settings(time_series,
                        neighbourhood=FixedRadius(epsilon),
                        similarity_measure=EuclideanMetric(), theiler_corrector=1)
    result = RQAComputation.create(settings).run()
    return np.array([result.recurrence_rate,
                     result.determinism,
                     result.average_diagonal_line,
                     result.trapping_time,
                     result.longest_diagonal_line,
                     result.divergence,
                     result.entropy_diagonal_lines,
                     result.laminarity])


def false_nearest_neighbors(timeseries, max_dim=100, tau=1, R_tol=15, A_tol=2):
    """
    Compute the False Nearest Neighbors to estimate the embedding dimension.
    
    Parameters:
    - timeseries: numpy array, the input time series
    - max_dim: int, maximum embedding dimension to test (default: 10)
    - tau: int, time delay (default: 1)
    - R_tol: float, distance tolerance (default: 15)
    - A_tol: float, relative size tolerance (default: 2)
    
    Returns:
    - optimal_dim: int, the estimated optimal embedding dimension
    """
    N = len(timeseries)
    fnn_fractions = []

    for dim in range(1, max_dim):
        # Construct delay embedding
        embed_len = N - (dim * tau)
        if embed_len <= 0:
            print(f"Warning: Embedding dimension {dim} is too large for the given time series length.")
            break
        
        embed = np.array([timeseries[i:i + dim * tau:tau] for i in range(embed_len)])
        
        # Find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(embed)
        distances, indices = nbrs.kneighbors(embed)
        
        # Compute FNN
        fnn_count = 0
        valid_points = 0
        for i in range(embed_len):
            d_curr = distances[i, 1]
            nn_curr = indices[i, 1]
            
            if d_curr == 0 or i + dim * tau >= N or nn_curr + dim * tau >= N:
                continue
            
            valid_points += 1
            
            # Check if the nearest neighbor is a false neighbor
            R_d = abs(timeseries[i + dim * tau] - timeseries[nn_curr + dim * tau]) / d_curr
            A_d = np.sqrt(R_d**2 + d_curr**2) / d_curr
            
            if R_d > R_tol or A_d > A_tol:
                fnn_count += 1
        
        fnn_fraction = fnn_count / valid_points if valid_points > 0 else 0
        fnn_fractions.append(fnn_fraction)
        
        print(f"Dimension {dim}: FNN fraction = {fnn_fraction:.4f}")
        
        # Check if FNN fraction is below threshold
        if fnn_fraction < 0.1:  # You can adjust this threshold
            return dim
    
    # If no clear dimension is found, return the dimension with the lowest FNN fraction
    optimal_dim = np.argmin(fnn_fractions) + 1
    print(f"No clear optimal dimension found. Returning dimension with lowest FNN fraction: {optimal_dim}")
    return optimal_dim