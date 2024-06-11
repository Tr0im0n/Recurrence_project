# Source for lorenz system: https://matplotlib.org/stable/gallery/mplot3d/lorenz_attractor.html

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform


def lorenz(xyz, *, s=10, r=28, b=2.667):
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])

def get_xyzs_lorenz():
    dt = 0.01
    num_steps = 10000

    xyzs = np.empty((num_steps + 1, 3))  # Need one more for the initial values
    xyzs[0] = (1., 1., 1.)  # Set initial values
    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps):
        xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt

    # Plot
    ax = plt.figure().add_subplot(projection='3d')

    ax.plot(*xyzs.T, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")

    plt.show()

    return(xyzs)


def RP_one_coord(xyzs, coord=0):
    data = xyzs[:,coord]

    epsilon = 1
    m = 5
    T = 5

    num_vectors = len(data) - (m - 1) * T
    vectors = np.array([data[t:t + m * T:T] for t in range(num_vectors)])
    D = squareform(pdist(vectors, metric='euclidean'))
    hit = np.argwhere(D < epsilon)

    x_rec, y_rec = hit[:, 0], hit[:, 1]

    # Plot points of recurrence
    plt.figure(figsize=(10, 8))
    plt.scatter(x_rec, y_rec, s=1)
    plt.title('Recurrence Plot')
    plt.xlabel('Vector Index')
    plt.ylabel('Vector Index')
    plt.show()

xyzs = get_xyzs_lorenz()
# RP_one_coord(xyzs, 0)

num_vectors = np.shape(xyzs)[0]
D = squareform(pdist(xyzs, metric='euclidean'))

# set epsilon to 10% of max phase space diameter
D_max = np.max(D)
epsilon = 0.1*D_max

# Find points of recurrence
hit = np.argwhere(D < epsilon)

# Extract x and y coordinates of points of recurrence
x_rec, y_rec = hit[:, 0], hit[:, 1]

# Plot the distance matrix D
plt.figure(figsize=(10, 8))
plt.imshow(D, origin='lower', cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('Distance Matrix')
plt.xlabel('Vector Index')
plt.ylabel('Vector Index')
# plt.show()

# Plot points of recurrence
plt.figure(figsize=(10, 8))
plt.scatter(x_rec, y_rec, s=1)
plt.title('Recurrence Plot')
plt.xlabel('Vector Index')
plt.ylabel('Vector Index')
plt.show()


# Another method of making RP for lorenz system... very different result.
# from scipy.integrate import solve_ivp
#
# # Lorenz system parameters
# sigma = 10.0
# beta = 8.0 / 3.0
# rho = 28.0
#
# def lorenz(t, state):
#     x, y, z = state
#     return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
#
# # Initial conditions
# initial_state = [1.0, 1.0, 1.0]
# t_span = [0, 25]
# t_eval = np.linspace(t_span[0], t_span[1], 1000)
#
# # Solve Lorenz system
# sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)
# X = sol.y.T  # Solution array
#
# # Compute distance matrix
# D = squareform(pdist(X, metric='euclidean'))
#
# # Define a threshold
# epsilon = 2.0
#
# # Create recurrence matrix
# R = (D <= epsilon).astype(int)
#
# # Plot the recurrence plot
# plt.imshow(R, cmap='binary', origin='lower')
# plt.xlabel('Time i')
# plt.ylabel('Time j')
# plt.title('Recurrence Plot for Lorenz System')
# plt.show()

