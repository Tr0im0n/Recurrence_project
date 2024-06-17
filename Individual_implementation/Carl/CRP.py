import numpy as np
from scipy.spatial.distance import cdist
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Generalized function to generate Cross Recurrence Plot
def CRP(ts1, ts2, m, T, epsilon):
    # Create trajectory matrices for both time series
    H_ts1 = np.array([ts1[i:i+m*T:T] for i in range(len(ts1) - m*T + 1)])
    H_ts2 = np.array([ts2[i:i+m*T:T] for i in range(len(ts2) - m*T + 1)])

    # Calculate pairwise distances between trajectory matrices
    distance_matrix = cdist(H_ts1, H_ts2, 'euclidean')

    # Form cross recurrence matrix based on the epsilon threshold
    cross_recurrence_matrix = (distance_matrix <= epsilon).astype(int)
    return cross_recurrence_matrix

# Example usage of the function with simulated Lorenz and Rössler systems
def simulate_system(func, params, initial_state, t_span, t_eval):
    sol = solve_ivp(func, t_span, initial_state, args=params, t_eval=t_eval, method='RK45')
    return sol.y[0]  # Returning the first component for simplicity

# Differential equations for the Lorenz system
def lorenz(t, Y, sigma, rho, beta):
    x, y, z = Y
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

# Differential equations for the Rössler system
def roessler(t, Y, a, b, c):
    x, y, z = Y
    return [-y - z, x + a * y, b + z * (x - c)]

# Parameters
sigma, rho, beta = 10, 28, 8/3
a, b, c = 0.2, 0.2, 5.7
initial_state = [1.0, 1.0, 1.0]
t_span = (0, 40)
t_eval = np.linspace(t_span[0], t_span[1], num=1000)
m, T, epsilon = 2, 1, 10  # Embedding dimension, time delay, and distance threshold

# Simulate systems
lorenz_series = simulate_system(lorenz, (sigma, rho, beta), initial_state, t_span, t_eval)
roessler_series = simulate_system(roessler, (a, b, c), initial_state, t_span, t_eval)

# Generate CRP
crp = CRP(lorenz_series, roessler_series, m, T, epsilon)

# Plotting
plt.figure(figsize=(8, 8))
plt.imshow(crp, cmap='Greys', origin='lower')
plt.xlabel('Rössler System Index')
plt.ylabel('Lorenz System Index')
plt.title('Cross Recurrence Plot Between Lorenz and Rössler Systems')
plt.show()
