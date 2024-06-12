import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby

# Generate a simple time series data
time_series = np.sin(np.linspace(0, 4 * np.pi, 100))  # Sine wave

# Parameters
threshold = 0.15  # Distance threshold for recurrence

# Create an empty recurrence matrix
recurrence_matrix = np.zeros((len(time_series), len(time_series)))

# Fill the recurrence matrix using nested for loops
for i in range(len(time_series)):
    for j in range(len(time_series)):
        distance = abs(time_series[i] - time_series[j])
        if distance < threshold:
            recurrence_matrix[i, j] = 1

# Plot the recurrence matrix
plt.imshow(recurrence_matrix, cmap='binary', origin='lower')
plt.title('Recurrence Plot')
plt.show()

#RQA with chatGPT

# Calculate recurrence rate (RR)
RR = np.sum(recurrence_matrix) / (len(time_series) ** 2)

# Calculate diagonal line structures
diagonals = [np.diag(recurrence_matrix, k) for k in range(-len(time_series) + 1, len(time_series))]
diag_lengths = [len(list(group)) for diag in diagonals for k, group in groupby(diag) if k == 1]

# Calculate DET
DET = sum(l for l in diag_lengths if l >= 2) / np.sum(recurrence_matrix) if np.sum(recurrence_matrix) != 0 else 0

# Calculate L
L = np.mean([l for l in diag_lengths if l >= 2]) if diag_lengths else 0

# Calculate Lmax
Lmax = max(diag_lengths) if diag_lengths else 0

# Calculate DIV
DIV = 1 / Lmax if Lmax != 0 else 0

# Calculate ENTR
counts = np.bincount(diag_lengths)
probs = counts / np.sum(counts)
ENTR = -np.sum(probs * np.log(probs)) if np.sum(counts) > 0 else 0

# Calculate trend (TREND)
TREND = np.mean([np.mean(recurrence_matrix[i, i:]) for i in range(len(recurrence_matrix))])

# Calculate laminarity (LAM)
verticals = [recurrence_matrix[:, i] for i in range(len(time_series))]
vert_lengths = [len(list(group)) for vert in verticals for k, group in groupby(vert) if k == 1]
LAM = sum(l for l in vert_lengths if l >= 2) / np.sum(recurrence_matrix) if np.sum(recurrence_matrix) != 0 else 0

# Calculate trapping time (TT)
TT = np.mean([l for l in vert_lengths if l >= 2]) if vert_lengths else 0

# Calculate maximum length of vertical structures (Vmax)
Vmax = max(vert_lengths) if vert_lengths else 0

# Calculate entropy of vertical structures (VENTR)
vert_counts = np.bincount(vert_lengths)
vert_probs = vert_counts / np.sum(vert_counts)
VENTR = -np.sum(vert_probs * np.log(vert_probs)) if np.sum(vert_counts) > 0 else 0

# Calculate mean recurrence time (MRT)
MRT = np.mean(np.diff(np.nonzero(recurrence_matrix)[0])) if np.sum(recurrence_matrix) > 0 else 0

# Calculate recurrence time entropy (RTE)
rt_diffs = np.diff(np.nonzero(recurrence_matrix)[0])
rt_counts = np.bincount(rt_diffs)
rt_probs = rt_counts / np.sum(rt_counts)
RTE = -np.sum(rt_probs * np.log(rt_probs)) if np.sum(rt_counts) > 0 else 0

# Calculate number of the most probable recurrence time (NMPRT)
NMPRT = np.argmax(rt_counts) if len(rt_counts) > 0 else 0

# Output the calculated RQA measures
print(f"RR (Recurrence Rate): {RR}")
print(f"DET (Determinism): {DET}")
print(f"L (Average Length of Diagonal Structures): {L}")
print(f"Lmax (Maximum Length of Diagonal Structures): {Lmax}")
print(f"DIV (Divergence): {DIV}")
print(f"ENTR (Entropy of Diagonal Structures): {ENTR}")
print(f"TREND (Trend of Recurrences): {TREND}")
print(f"LAM (Laminarity): {LAM}")
print(f"TT (Trapping Time): {TT}")
print(f"Vmax (Maximum Length of Vertical Structures): {Vmax}")
print(f"VENTR (Entropy of Vertical Structures): {VENTR}")
print(f"MRT (Mean Recurrence Time): {MRT}")
print(f"RTE (Recurrence Time Entropy): {RTE}")
print(f"NMPRT (Number of the Most Probable Recurrence Time): {NMPRT}")
