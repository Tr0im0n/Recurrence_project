import numpy as np
import matplotlib.pyplot as plt
from setuptools import sic

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
plt.xlabel('Time')
plt.ylabel('Time')
plt.show()
