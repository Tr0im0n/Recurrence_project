import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.distance import cdist
import pyunicorn

# Differential equations for the Lorenz system
def lorenz(t, Y, sigma, rho, beta):
    x, y, z = Y
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

# Differential equations for the Rössler system
def roessler(t, Y, a, b, c):
    x, y, z = Y
    return [-y - z, x + a * y, b + z * (x - c)]

# Function to simulate a system using solve_ivp
def simulate_system(func, params, initial_state, t_span, t_eval):
    sol = solve_ivp(func, t_span, initial_state, args=params, t_eval=t_eval, method='RK45')
    return sol.y[0]  # Returning the first component for simplicity

# Generate Cross Recurrence Plot
def generateCRP(sigma, rho, beta, a, b, c, m, T, epsilon):
    # Simulation settings
    initial_state = [1.0, 1.0, 1.0]
    t_span = (0, 40)
    t_eval = np.linspace(t_span[0], t_span[1], num=1000)

    # Simulate Lorenz and Rössler systems
    lorenz_series = simulate_system(lorenz, (sigma, rho, beta), initial_state, t_span, t_eval)
    roessler_series = simulate_system(roessler, (a, b, c), initial_state, t_span, t_eval)

    # Create trajectory matrices
    H_lorenz = np.array([lorenz_series[i:i+m*T:T] for i in range(len(lorenz_series) - m*T + 1)])
    H_roessler = np.array([roessler_series[i:i+m*T:T] for i in range(len(roessler_series) - m*T + 1)])

    # Calculate pairwise distances
    distance_matrix = cdist(H_lorenz, H_roessler, 'euclidean')

    # Form cross recurrence matrix
    cross_recurrence_matrix = (distance_matrix <= epsilon).astype(int)
    return cross_recurrence_matrix

# Parameters for the CRP
sigma, rho, beta = 10, 28, 8/3
a, b, c = 0.2, 0.2, 5.7
m, T, epsilon = 2, 1, 10  # Example values for embedding dimension, time delay, and distance threshold

# Generate and visualize the Cross Recurrence Plot
crp = generateCRP(sigma, rho, beta, a, b, c, m, T, epsilon)

# Plotting
plt.figure(figsize=(8, 8))
plt.imshow(crp, cmap='Greys', origin='lower')
plt.xlabel('Rössler System Index')
plt.ylabel('Lorenz System Index')
plt.title('Cross Recurrence Plot Between Lorenz and Rössler Systems')
plt.show()