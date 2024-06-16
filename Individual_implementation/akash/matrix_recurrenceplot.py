import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hankel

# Define the sequence
sequence = np.array([1, 2, 3, 4,8])

# Construct the Hankel matrix
H = hankel(sequence)
#print("Hankel Matrix H:\n", H)

# Compute the Kronecker product
H_kron = np.kron(H, np.ones(H.shape)) - np.kron(np.ones(H.shape), H)
##print("Kronecker Product H_kron:\n", H_kron)

# Calculate row-wise L2 norms
row_norms = np.linalg.norm(H_kron, axis=1)
#print("Row-wise norms:\n", row_norms)

# Define the threshold epsilon
epsilon = 1.0

# Construct the recurrence matrix
n = len(row_norms)
R = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        R[i, j] = 1 if np.abs(row_norms[i] - row_norms[j]) <= epsilon else 0

print("Recurrence Matrix R:\n", R)

# Plot the recurrence matrix
plt.imshow(R, cmap='binary', origin='lower')
plt.title('Recurrence Plot')
plt.xlabel('Time Index')
plt.ylabel('Time Index')
plt.show()
