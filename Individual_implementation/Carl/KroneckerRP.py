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

def reshape(distance_vector):
    distance_matrix = distance_vector.reshape((3, -1))
    return distance_matrix

# Generate time series data
t_span = (0, 40)
t_eval = np.linspace(t_span[0], t_span[1], num=100)
sol = solve_ivp(lorenz, t_span, initial_state, args=(sigma, rho, beta), t_eval=t_eval, method='RK45')


timeseries = sol.y[0]
l = timeseries.shape[0]
ones = np.ones_like(timeseries)

m = 3 # embedding dimension
T = 1 # delay
epsilon = 0.1 # threshold

H = hankel(timeseries[:l - m * T], timeseries[m * T:])
P = np.kron(ones, H) - np.kron(H, ones)
distance_matrix = squareform(pdist(P, 'euclidean'))

recurrence_matrix = (distance_matrix <= epsilon).astype(int)

#Plot the recurrence matrix
plt.imshow(recurrence_matrix, cmap='binary', origin='lower')
plt.title('Recurrence Plot')
plt.xlabel('Time Steps')
plt.ylabel('Time Steps')
plt.show()
