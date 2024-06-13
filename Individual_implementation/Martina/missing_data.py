import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Example time series with missing data
time_series = np.array([1, 2, 3, np.nan, 5, 6, np.nan, 8, 9, 10])
embedding_dimension = 2
time_delay = 1
epsilon = 0.5

# Function to create embedded vectors
def embed_time_series(time_series, embedding_dimension, time_delay):
    N = len(time_series)
    embedded_vectors = []
    for i in range(N - (embedding_dimension - 1) * time_delay):
        if not np.isnan(time_series[i:i + embedding_dimension * time_delay:time_delay]).any():
            embedded_vectors.append(time_series[i:i + embedding_dimension * time_delay:time_delay])
    return np.array(embedded_vectors)

# Embedding the time series
embedded_vectors = embed_time_series(time_series, embedding_dimension, time_delay)

# Calculate distance matrix and create recurrence plot
distance_matrix = squareform(pdist(embedded_vectors))
recurrence_plot = distance_matrix < epsilon

# Function to predict missing data based on local recurrence structure
def predict_missing_data(time_series, recurrence_plot, embedding_dimension, time_delay):
    for i in range(len(time_series)):
        if np.isnan(time_series[i]):
            # Find nearest neighbors in the embedded space
            neighbors = []
            for j in range(len(recurrence_plot)):
                if recurrence_plot[j].any():
                    neighbors.append(time_series[j:j + embedding_dimension * time_delay:time_delay])
            # Average the neighbors to predict the missing value
            if neighbors:
                time_series[i] = np.nanmean(neighbors)
    return time_series

# Predict missing data points
filled_time_series = predict_missing_data(time_series, recurrence_plot, embedding_dimension, time_delay)

print("Filled Time Series:")
print(filled_time_series)
