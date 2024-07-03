import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from pyts.image import RecurrencePlot

# def recurrence_plot_no_embedding(data, epsilon, title):
#     D = squareform(pdist(data.reshape(-1, 1), metric='euclidean'))
#     R = (D <= epsilon).astype(int)
#     plt.figure(figsize=(6, 6))
#     plt.imshow(R, cmap='binary', origin='lower')
#     plt.title(f'{title} (no embedding)')
#     plt.xlabel('Time i')
#     plt.ylabel('Time j')
#     plt.show()
#
# def recurrence_plot_embedding(data, epsilon, title, m, T):
#     num_vectors = len(data) - (m - 1) * T
#     vectors = np.array([data[t:t + m * T:T] for t in range(num_vectors)])
#
#     D = squareform(pdist(vectors, metric='euclidean'))
#     R = (D <= epsilon).astype(int)
#     plt.figure(figsize=(6, 6))
#     plt.imshow(R, cmap='binary', origin='lower')
#     plt.title(f'{title} (embedding)')
#     plt.xlabel('Time i')
#     plt.ylabel('Time j')
#     plt.show()
#
# # Homogeneous: Random data
# data_homogeneous = np.random.rand(1000)
# recurrence_plot_no_embedding(data_homogeneous, epsilon=0.1, title='Homogeneous Recurrence Plot')
# recurrence_plot_embedding(data_homogeneous, epsilon=0.8, title='Homogeneous Recurrence Plot', m=5, T=2)
#
#
# # Periodic: Sine wave data
# t = np.linspace(0, 4 * np.pi, 1000)
# data_periodic = np.sin(t)
# recurrence_plot_no_embedding(data_periodic, epsilon=0.05, title='Periodic Recurrence Plot')
# recurrence_plot_embedding(data_periodic, epsilon=0.05, title='Periodic Recurrence Plot', m=5, T=2)
#
# # Drift: Linearly increasing data
# drift_rate = 0.25
# t = np.linspace(0, 10, 1000)
# # data_drift = (np.sin(2 * np.pi * t) + 0.5 * np.sin(4 * np.pi * t) + drift_rate * t)
# data_drift = (np.sin(2 * np.pi * t) + drift_rate * t)
# # data_drift = (np.sin(2 * np.pi * t))
# recurrence_plot_no_embedding(data_drift, epsilon=0.1, title='Drift Recurrence Plot')
# recurrence_plot_embedding(data_drift, epsilon=2, title='Drift Recurrence Plot', m=12, T=6)
#
# # Laminar: Piecewise constant data
# data_laminar = np.concatenate([np.ones(250), np.ones(250) * 2, np.ones(250) * 3, np.ones(250) * 4]).reshape(-1, 1)
# # recurrence_plot_no_embedding(data_laminar, epsilon=0.5, title='Laminar Recurrence Plot')
#
# # White noise
# white_noise = np.random.normal(size=500)
# recurrence_plot_no_embedding(white_noise, epsilon=0.5, title='White Noise Recurrence Plot')
# recurrence_plot_embedding(white_noise, epsilon=2, title='White Noise Recurrence Plot', m=5, T=2)

#______________________________________________________________________________
# # Generate a 10 Hz s ine wave with a sampling rate of 1000 Hz (100 ms wavelength)
# sampling_rate = 1000  # Hz
# frequency = 10  # Hz
# duration = 1  # second
# t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
# sine_wave = np.sin(2 * np.pi * frequency * t)
#
#
# # Parameters for recurrence plot
# embedding_dimension = 10
# time_delay = 10                     # Why does time delay equal to period not give issues?
# fixed_recurrence_rate = 0.1  # 10%
#
# # Create embedded vectors
# def create_embedded_vectors(data, dimension, delay):
#     num_vectors = len(data) - (dimension - 1) * delay
#     vectors = np.array([data[i:i + dimension * delay:delay] for i in range(num_vectors)])
#     return vectors
#
# embedded_vectors = create_embedded_vectors(sine_wave, embedding_dimension, time_delay)
#
# # Compute distance matrix
# D = squareform(pdist(embedded_vectors, metric='euclidean'))
#
# # Determine the threshold epsilon for the fixed recurrence rate
# sorted_distances = np.sort(D.flatten())
# threshold_index = int(fixed_recurrence_rate * len(sorted_distances))
# epsilon = sorted_distances[threshold_index]
#
# # Create recurrence matrix
# R = (D <= epsilon).astype(int)
#
# # Plot the sine wave
# plt.figure(figsize=(10, 4))
# plt.plot(t, sine_wave)
# plt.title('10 Hz Sine Wave')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.show()
#
# # Plot the recurrence plot
# plt.figure(figsize=(6, 6))
# plt.imshow(R, cmap='binary', origin='lower')
# plt.title('Recurrence Plot (Embedding Dimension = 6, Time Delay = 6, RR = 10%)')
# plt.xlabel('Time Index')
# plt.ylabel('Time Index')
# plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.integrate import solve_ivp

def recurrence_plot(data, dimension, delay, epsilon):
    num_vectors = len(data) - (dimension - 1) * delay
    vectors = np.array([data[i:i + dimension * delay:delay] for i in range(num_vectors)])
    distance_matrix = squareform(pdist(vectors, 'euclidean'))
    distance_matrix_max = np.max(distance_matrix)
    distance_matrix_norm = distance_matrix / distance_matrix_max
    return distance_matrix_norm < epsilon

# Generate time series data
t_sine = np.linspace(0, 1, 500)
sine_wave = np.sin(2 * np.pi * 10 * t_sine)

t_white_noise = np.linspace(0, 1, 500)
white_noise = np.random.normal(0, 1, t_white_noise.shape)

def lorenz(t, state):
    x, y, z = state
    sigma = 10
    rho = 28
    beta = 8/3
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

t_lorenz = np.linspace(0, 50, 10000)
initial_state = [10, 10, 10]
sol = solve_ivp(lorenz, [t_lorenz[0], t_lorenz[-1]], initial_state, t_eval=t_lorenz)
lorenz_attractor = sol.y[0]

# Calculate recurrence plots
rp_sine = recurrence_plot(sine_wave, 10, 10, epsilon=0.1)
rp_white_noise = recurrence_plot(white_noise, 1, 1, epsilon=0.1)
rp_lorenz = recurrence_plot(lorenz_attractor, 1, 1, epsilon=0.1)

# Plot time series and recurrence plots
fig = plt.figure(figsize=(30, 10))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 3], width_ratios=[1, 1, 1])

# Sine wave
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(t_sine, sine_wave, color='blue')
# ax1.set_title('Sine Wave', fontsize=30)
# ax1.set_xlabel('Time [s]', fontsize=30)
# ax1.set_ylabel('Amplitude', fontsize=30)
ax1.tick_params(axis='both', which='major', labelsize=20)

# Recurrence plot for sine wave
ax2 = fig.add_subplot(gs[1, 0])
ax2.imshow(rp_sine, cmap='binary', origin='lower', extent=[0, 1, 0, 1])
# ax2.set_title('Recurrence Plot (Sine Wave)', fontsize=16)
# ax2.set_xlabel('Time [s]', fontsize=30)
# ax2.set_ylabel('Time [s]', fontsize=30)
ax2.tick_params(axis='both', which='major', labelsize=20)

# White noise
ax3 = fig.add_subplot(gs[0, 1])
ax3.plot(t_white_noise, white_noise, color='blue')
# ax3.set_title('White Noise', fontsize=16)
# ax3.set_xlabel('Time [s]', fontsize=30)
# ax3.set_ylabel('Amplitude', fontsize=30)
ax3.tick_params(axis='both', which='major', labelsize=20)

# Recurrence plot for white noise
ax4 = fig.add_subplot(gs[1, 1])
ax4.imshow(rp_white_noise, cmap='binary', origin='lower', extent=[0, 1, 0, 1])
# ax4.set_title('Recurrence Plot (White Noise)', fontsize=16)
# ax4.set_xlabel('Time [s]', fontsize=30)
# ax4.set_ylabel('Time [s]', fontsize=30)
ax4.tick_params(axis='both', which='major', labelsize=20)

# Lorenz attractor
ax5 = fig.add_subplot(gs[0, 2])
ax5.plot(t_lorenz, lorenz_attractor, color='blue')
# ax5.set_title('Lorenz Attractor', fontsize=16)
# ax5.set_xlabel('Time [s]', fontsize=30)
# ax5.set_ylabel('X Coordinate', fontsize=30)
ax5.tick_params(axis='both', which='major', labelsize=20)

# Recurrence plot for Lorenz attractor
ax6 = fig.add_subplot(gs[1, 2])
ax6.imshow(rp_lorenz, cmap='binary', origin='lower', extent=[0, 50, 0, 50])
# ax6.set_title('Recurrence Plot (Lorenz Attractor)', fontsize=16)
# ax6.set_xlabel('Time [s]', fontsize=30)
# ax6.set_ylabel('Time [s]', fontsize=30)
ax6.tick_params(axis='both', which='major', labelsize=20)


# plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.3, wspace=0.3)

plt.tight_layout()
plt.savefig('visualize_rp.png', dpi=500)

plt.show()