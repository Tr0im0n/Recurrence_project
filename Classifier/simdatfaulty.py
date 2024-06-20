import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Function to create single fault impulse signal with tapering off
def single_fault_impulse(t, f_osc, alpha):
    return np.exp(-alpha * t) * np.sin(2 * np.pi * f_osc * t)

# Generate fault impulses with noise, tapering off after 0.01 seconds
def generate_fault_impulses(fault_freq, sampling_freq, duration, f_osc, alpha, noise_level=0.05, increasing_rate=0.2):
    t = np.arange(0, duration, 1/sampling_freq)
    impulses = np.zeros_like(t)
    fault_array = np.zeros_like(t, dtype=int)
    period = 1 / fault_freq
    fault_times = []
    for i in range(int(duration * fault_freq)):
        start = int(i * period * sampling_freq)
        end = int(start + 0.01 * sampling_freq)
        # Increase the probability of fault occurrence over time
        if np.random.rand() < increasing_rate * (i / (duration * fault_freq)):
            impulses[start:end] = single_fault_impulse(np.linspace(0, 0.2, end - start), f_osc, alpha)
            fault_times.append(start / sampling_freq)
            fault_array[start:end] = 1
    noise = noise_level * np.random.normal(size=t.shape)
    return t, impulses + noise, fault_times, fault_array

# Generate normal vibration signal of healthy machine
def generate_healthy_vibration(sampling_freq, duration, base_freqs, amplitudes, noise_level=0.05):
    t = np.arange(0, duration, 1/sampling_freq)
    healthy_vibrations = np.zeros_like(t)
    for base_freq, amplitude in zip(base_freqs, amplitudes):
        healthy_vibrations += amplitude * np.sin(2 * np.pi * base_freq * t)
    noise = noise_level * np.random.normal(size=t.shape)
    return t, healthy_vibrations + noise

# Parameters
f_r = 30  # Rotating frequency (Hz)
f_osc = 1000  # Oscillating frequency of fault impulse (Hz)
alpha = 100  # Attenuation rate to ensure quick tapering
sampling_freq = 5000  # Sampling frequency (Hz)
duration = 5 # Duration of signal (seconds)
increasing_rate = 0.02  # Rate at which fault occurrence probability increases

# Estimate fault frequency for outer race
n = 8  # Number of rolling elements
D_p = 0.1  # Pitch diameter (meters)
d = 0.02  # Rolling element diameter (meters)
theta = np.pi / 6  # Contact angle (radians)
f_o = (n / 2) * (1 - d / D_p * np.cos(theta)) * f_r

# Generate fault impulses
t_fault, impulses_fault, fault_times, fault_array = generate_fault_impulses(f_o, sampling_freq, duration, f_osc, alpha, increasing_rate=increasing_rate)

# Generate normal vibration signal of healthy machine
base_freqs = [50, 150, 300]  # Example base frequencies of healthy vibrations
amplitudes = [1, 0.5, 0.2]   # Corresponding amplitudes
t_healthy, healthy_vibrations = generate_healthy_vibration(sampling_freq, duration, base_freqs, amplitudes)

# Combine healthy vibrations with fault impulses
combined_signal = healthy_vibrations + impulses_fault

# Plotting the vibrations over the first 0.1 seconds
time_limit = 5  # 0.1 seconds
indices = t_fault < time_limit

plt.figure(figsize=(12, 8))

# Plot with faults
plt.subplot(3, 1, 1)
plt.plot(t_fault[indices], combined_signal[indices])
plt.title('Simulated Fault Impulses with Healthy Vibrations')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
# Mark fault impulses with red dots
for fault_time in fault_times:
    if fault_time < time_limit:
        plt.plot(fault_time, single_fault_impulse(0, f_osc, alpha), 'ro')

# Plot without faults
plt.subplot(3, 1, 2)
plt.plot(t_healthy[indices], healthy_vibrations[indices])
plt.title('Healthy Machine Vibrations ')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot fault array
plt.subplot(3, 1, 3)
plt.plot(t_fault[indices], fault_array[indices])
plt.title('Fault Occurrences ')
plt.xlabel('Time (s)')
plt.ylabel('Fault Indicator (0 or 1)')

plt.tight_layout()
plt.show()

# Save synthetic data to CSV
synthetic_data = pd.DataFrame({
    'Time': t_fault,
    'Signal': combined_signal,
    'Fault': fault_array
})
file_path = 'synthetic_fault_data.csv'
synthetic_data.to_csv(file_path, index=False)