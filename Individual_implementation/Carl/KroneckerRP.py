import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import hankel
from scipy.spatial.distance import pdist, squareform

sigma, rho, beta = 10, 28, 8/3
initial_state = [1.0, 1.0, 1.0]
def lorenz(t, Y, sigma, rho, beta):
    x, y, z = Y
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

def fastRP(timeseries, m, T, epsilon):
    l = timeseries.shape[0]
    ones = np.ones_like(timeseries).T

    H = np.zeros((l-m+1, m)) # Trajectory Matrix
    for i in range(l-m*T+1):
        H[i] = timeseries[i:i+m*T:T]

    P = np.kron(ones, H) - np.kron(H, ones) 
    distance_matrix = squareform(pdist(P, 'euclidean'))

    recurrence_matrix = (distance_matrix <= epsilon).astype(int)
    return recurrence_matrix

# Generate time series data
t_span = (0, 40)
t_eval = np.linspace(t_span[0], t_span[1], num=100)
sol = solve_ivp(lorenz, t_span, initial_state, args=(sigma, rho, beta), t_eval=t_eval, method='RK45')

m = 5 # embedding dimension
T = 2 # delay
epsilon = 100 # threshold

timeseries = sol.y[0]

recurrence_matrix = fastRP(timeseries, m, T, epsilon)

#Plot the recurrence matrix
plt.imshow(recurrence_matrix, cmap='binary', origin='lower')
plt.title('Recurrence Plot')
plt.xlabel('Time Steps')
plt.ylabel('Time Steps')
plt.show()
