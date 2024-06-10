# Simple Implementation
import numpy as np
import matplotlib.pyplot as plt

signal = np.sin(np.linspace(0, 4 * np.pi, 100))

threshold = 0.05 
M = np.zeros((len(signal), len(signal)))

for i in range(len(signal)):
    for j in range(len(signal)):
        distance = abs(signal[i] - signal[j])
        if distance < threshold:
            M[i, j] = 1


plt.imshow(M, cmap='binary', origin='lower')
plt.title('Recurrence Plot')
plt.show()