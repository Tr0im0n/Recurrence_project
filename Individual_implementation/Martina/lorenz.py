# Source for lorenz system: https://matplotlib.org/stable/gallery/mplot3d/lorenz_attractor.html

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from pyts.image import RecurrencePlot
import time


def lorenz(xyz, *, s=10, r=28, b=2.667):
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])

def get_xyzs_lorenz(num_steps: int = 2500, dt: float = 0.01, show_plot: bool = True):
    xyzs = np.empty((num_steps + 1, 3))  # Need one more for the initial values
    xyzs[0] = (1., 1., 1.)  # Set initial values
    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps):
        xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt

    if not show_plot:
        return xyzs
    # Plot
    ax = plt.figure().add_subplot(projection='3d')

    ax.plot(*xyzs.T, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")

    # plt.show()

    return(xyzs)


def RP_pyts_one_coord(xyzs, coord=0):
    start_time = time.perf_counter()
    data = xyzs[:,coord].reshape(1, -1)

    rp = RecurrencePlot()
    data_rp = rp.transform(data)

    # Plot the distance matrix D
    plt.figure(figsize=(6, 6))
    plt.imshow(data_rp[0], cmap='binary', origin='lower')
    plt.title('Recurrence Plot')


    plt.show()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print('method pyts: ', elapsed_time)

    # # Plot points of recurrence
    # plt.figure(figsize=(10, 8))
    # plt.scatter(x_rec, y_rec, s=1)
    # plt.title('Recurrence Plot')
    # plt.xlabel('Vector Index')
    # plt.ylabel('Vector Index')
    # plt.show()

def RP_one_coord(xyzs, coord=0):
    start_time = time.perf_counter()
    data = xyzs[:,coord]

    epsilon = 0.1
    m = 1
    T = 1

    num_vectors = len(data) - (m - 1) * T
    vectors = np.array([data[t:t + m * T:T] for t in range(num_vectors)])
    D = squareform(pdist(vectors, metric='euclidean'))
    D_max = np.max(D)
    D_norm = D / D_max
    recurrence_matrix = D_norm < epsilon
    hit = np.argwhere(D_norm < epsilon)

    x_rec, y_rec = hit[:, 0], hit[:, 1]

    # Plot the distance matrix D
    plt.figure(figsize=(10, 8))
    plt.imshow(recurrence_matrix, origin='lower', cmap='binary', interpolation='nearest')
    # plt.colorbar()
    plt.title('Distance Matrix')
    plt.xlabel('Vector Index')
    plt.ylabel('Vector Index')
    plt.savefig('lorenz_2500_highres', dpi=2000)
    plt.show()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print('method manual: ', elapsed_time)

    # # Plot points of recurrence
    # plt.figure(figsize=(10, 8))
    # plt.scatter(x_rec, y_rec, s=1)
    # plt.title('Recurrence Plot')
    # plt.xlabel('Vector Index')
    # plt.ylabel('Vector Index')
    # plt.show()


if __name__ == "__main__":
    xyzs = get_xyzs_lorenz()
    RP_pyts_one_coord(xyzs, 0)
    RP_one_coord(xyzs, 0)


