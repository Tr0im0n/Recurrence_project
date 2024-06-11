# Source for lorenz system: https://matplotlib.org/stable/gallery/mplot3d/lorenz_attractor.html

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform


def lorenz(xyz, *, s=10, r=28, b=2.667):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])

def get_xyzs_lorenz():
    dt = 0.01
    num_steps = 10000

    xyzs = np.empty((num_steps + 1, 3))  # Need one more for the initial values
    xyzs[0] = (0., 1., 1.05)  # Set initial values
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

xyzs = get_xyzs_lorenz()

num_vectors = np.shape(xyzs)[0]
D = squareform(pdist(xyzs, metric='euclidean'))

epsilon = 3

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
plt.show()

# Plot points of recurrence
plt.figure(figsize=(10, 8))
plt.scatter(x_rec, y_rec, s=1)
plt.title('Recurrence Plot')
plt.xlabel('Vector Index')
plt.ylabel('Vector Index')
plt.show()





