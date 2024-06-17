import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.linalg import hankel
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

def getexampledata():
    sigma, rho, beta = 10, 28, 8/3
    initial_state = [1.0, 1.0, 1.0]
    t_span = (0, 40)
    t_eval = np.linspace(t_span[0], t_span[1], num=1000)
    sol = solve_ivp(lorenz, t_span, initial_state, args=(sigma, rho, beta), t_eval=t_eval, method='RK45')
    return sol.y[0]
def lorenz(t, Y, sigma, rho, beta):
    x, y, z = Y
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

def RecurrencePlot(timeseries, m, T, epsilon):
    l = timeseries.shape[0]
    ones = np.ones_like(timeseries)

    H = np.zeros((l-m+1, m)) # Trajectory Matrix
    for i in range(l-m*T+1):
        H[i] = timeseries[i:i+m*T:T]

    P = np.kron(ones, H) - np.kron(H, ones) 
    distance_matrix = squareform(pdist(P, 'euclidean'))

    recurrence_matrix = (distance_matrix <= epsilon).astype(int)
    return recurrence_matrix

def min_embedding_dim(timeseries, max_dim, T=1, Rtol=10.0, Atol=2.0):
    n = len(timeseries)

    for d in range(1, max_dim + 1):
        embedding = np.zeros((n - d * T + 1, d))
        for i in range(n - d * T + 1):
            embedding[i] = timeseries[i:i + d * T:T]

        if embedding.shape[0] < 2:
            continue  # Need at least two points to proceed

        # Find nearest neighbors in the current dimension
        nn = NearestNeighbors(n_neighbors=2).fit(embedding)
        distances, indices = nn.kneighbors(embedding)

        count_fnn = 0
        for i in range(len(embedding)):
            dist_d = distances[i, 1]
            neighbor_idx = indices[i, 1]

            # Check next higher dimension
            if i + d * T < n and neighbor_idx + d * T < n:
                point_d1 = np.append(embedding[i], timeseries[i + d * T])
                neighbor_point_d1 = np.append(embedding[neighbor_idx], timeseries[neighbor_idx + d * T])
                dist_d1 = np.linalg.norm(point_d1 - neighbor_point_d1)

                if dist_d1 / dist_d > Rtol or abs(dist_d1 - dist_d) > Atol:
                    count_fnn += 1

        fnn_ratio = count_fnn / len(embedding)
        if fnn_ratio < 0.1:
            return d  # Return the current dimension as the minimum

    return -1  # Return -1 if no suitable dimension is found

timeseries = getexampledata()
m = 4 # embedding dimension
T = 2 # delay
epsilon = 75 # threshold

m = min_embedding_dim(timeseries, max_dim=50, T=T)

recurrence_matrix = RecurrencePlot(timeseries, m, T, epsilon)

#Plot the recurrence matrix
plt.imshow(recurrence_matrix, cmap='binary', origin='lower')
plt.title('Recurrence Plot')
plt.xlabel('Time Steps')
plt.ylabel('Time Steps')
plt.show()
