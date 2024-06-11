import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from pyts.image import RecurrencePlot

def plot_recurrence_plot(data, epsilon, title):
    D = squareform(pdist(data, metric='euclidean'))
    R = (D <= epsilon).astype(int)
    plt.figure(figsize=(6, 6))
    plt.imshow(R, cmap='binary', origin='lower')
    plt.title(title)
    plt.xlabel('Time i')
    plt.ylabel('Time j')
    plt.show()

# Homogeneous: Random data
data_homogeneous = np.random.rand(1000, 1)
# plot_recurrence_plot(data_homogeneous, epsilon=0.1, title='Homogeneous Recurrence Plot')

# Periodic: Sine wave data
t = np.linspace(0, 4 * np.pi, 1000)
data_periodic = np.sin(t).reshape(-1, 1)
# plot_recurrence_plot(data_periodic, epsilon=0.05, title='Periodic Recurrence Plot')

# Drift: Linearly increasing data
data_drift = np.linspace(0, 10, 1000).reshape(-1, 1)
# plot_recurrence_plot(data_drift, epsilon=1, title='Drift Recurrence Plot')

# Laminar: Piecewise constant data
data_laminar = np.concatenate([np.ones(250), np.ones(250) * 2, np.ones(250) * 3, np.ones(250) * 4]).reshape(-1, 1)
# plot_recurrence_plot(data_laminar, epsilon=0.5, title='Laminar Recurrence Plot')

#______________________________________________________________________________
# Generate a 10 Hz sine wave with a sampling rate of 1000 Hz (100 ms wavelength)
sampling_rate = 1000  # Hz
frequency = 10  # Hz
duration = 1  # second
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
sine_wave = np.sin(2 * np.pi * frequency * t)


# Parameters for recurrence plot
embedding_dimension = 10
time_delay = 10
fixed_recurrence_rate = 0.1  # 10%

# Create embedded vectors
def create_embedded_vectors(data, dimension, delay):
    num_vectors = len(data) - (dimension - 1) * delay
    vectors = np.array([data[i:i + dimension * delay:delay] for i in range(num_vectors)])
    return vectors

embedded_vectors = create_embedded_vectors(sine_wave, embedding_dimension, time_delay)

# Compute distance matrix
D = squareform(pdist(embedded_vectors, metric='euclidean'))

# Determine the threshold epsilon for the fixed recurrence rate
sorted_distances = np.sort(D.flatten())
threshold_index = int(fixed_recurrence_rate * len(sorted_distances))
epsilon = sorted_distances[threshold_index]

# Create recurrence matrix
R = (D <= epsilon).astype(int)

# Plot the sine wave
plt.figure(figsize=(10, 4))
plt.plot(t, sine_wave)
plt.title('10 Hz Sine Wave')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Plot the recurrence plot
plt.figure(figsize=(6, 6))
plt.imshow(R, cmap='binary', origin='lower')
plt.title('Recurrence Plot (Embedding Dimension = 6, Time Delay = 6, RR = 10%)')
plt.xlabel('Time Index')
plt.ylabel('Time Index')
plt.show()
