import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# NOT SURE ABOUT THE MATH HERE


def generateRP(time_series, m, T, epsilon):    
    # Check if the time series is univariate and adjust shape accordingly
    if time_series.ndim == 1:
        time_series = time_series[:, np.newaxis]

    # Number of points in the time series
    n_features, n_points = time_series.shape

    # Create the trajectory matrix using embedding dimension and time delay
    trajectory_matrix = np.zeros((n_points - (m - 1) * T, m * n_features))
    for i in range(m):
        trajectory_matrix[:, i*n_features:(i+1)*n_features] = time_series[:, i*T:i*T + n_points - (m - 1) * T].T

    # Calculate pairwise distances using the norm
    distance_matrix = np.linalg.norm(trajectory_matrix[:, np.newaxis, :] - trajectory_matrix[np.newaxis, :, :], axis=2)

    # Generate the recurrence matrix
    recurrence_matrix = (distance_matrix <= epsilon).astype(int)
    return recurrence_matrix

def lorenz(t, Y, sigma, rho, beta):
    x, y, z = Y
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

def simulate_system(func, params, initial_state, t_span, t_eval):
    sol = solve_ivp(func, t_span, initial_state, args=params, t_eval=t_eval, method='RK45')
    return sol.y

# Example usage
m, T, epsilon = 10, 2, 50  # Example parameters

# Simulate the Lorenz system
sigma, rho, beta = 10, 28, 8/3
initial_state = [1.0, 1.0, 1.0]
t_span = (0, 40)
t_eval = np.linspace(t_span[0], t_span[1], num=1000)
lorenz_data = simulate_system(lorenz, (sigma, rho, beta), initial_state, t_span, t_eval)

recurrence_matrix = generateRP(lorenz_data, m, T, epsilon)

plt.figure(figsize=(6, 6))
plt.imshow(recurrence_matrix, cmap='Greys', origin='lower')
plt.title('Recurrence Plot')
plt.xlabel('Time')
plt.ylabel('Time')
plt.show()
