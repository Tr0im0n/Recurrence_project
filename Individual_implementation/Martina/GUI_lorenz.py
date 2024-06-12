import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
from scipy.spatial.distance import pdist, squareform

class LivePlotAppSine:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Sine Wave Plot")

        # Create a figure and a subplot for the sine wave
        self.fig = plt.Figure(figsize=(10, 5))
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)

        # Embedding the Matplotlib figure into Tkinter canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Add Start and Stop buttons to control the live plotting
        self.btn_start = ttk.Button(root, text="Start", command=self.start)
        self.btn_start.pack(side=tk.LEFT)

        self.btn_stop = ttk.Button(root, text="Stop", command=self.stop)
        self.btn_stop.pack(side=tk.RIGHT)

        # Variables to control the live plotting
        self.is_running = False
        self.time_step = 0
        self.dt = 0.1
        self.x_data = np.linspace(0, 4 * np.pi, 100)  # Fixed x data for sine wave
        self.y_data = np.sin(self.x_data)  # Initial y data for sine wave
        self.freq = 1  # Initial frequency
        self.freq_change_interval = 1000  # Change frequency every 1000 ms
        self.change_frequency()

    def update_plot(self):
        if not self.is_running:
            return
        self.time_step += self.dt
        self.y_data = np.sin(self.x_data * self.freq + self.time_step)
        self.ax1.clear()
        self.ax1.plot(self.x_data, self.y_data, lw=2)
        self.ax1.set_title("Live Sine Wave")
        self.ax1.set_xlabel("X Axis")
        self.ax1.set_ylabel("sin(X + t)")

        self.update_recurrence_plot()

        self.canvas.draw()
        self.root.after(100, self.update_plot)

    def change_frequency(self):
        if self.is_running:
            self.freq = random.uniform(0.5, 2.0)  # Random frequency between 0.5 and 2.0
            self.root.after(self.freq_change_interval, self.change_frequency)

    def update_recurrence_plot(self):
        self.ax2.clear()

        m = 10
        T = 3
        num_vectors = len(self.y_data) - (m - 1) * T
        vectors = np.array([self.y_data[t:t + m * T:T] for t in range(num_vectors)])

        D = np.zeros((num_vectors, num_vectors))
        for i in range(num_vectors):
            for j in range(num_vectors):
                D[i, j] = np.linalg.norm(vectors[i] - vectors[j])

        epsilon = 0.5
        hit = np.argwhere(D < epsilon)
        x_rec, y_rec = hit[:, 0], hit[:, 1]

        self.ax2.scatter(x_rec, y_rec, s=1)
        self.ax2.set_title("Recurrence Plot")
        self.ax2.set_xlabel("Vector Index")
        self.ax2.set_ylabel("Vector Index")

    def start(self):
        if self.is_running:
            return
        print('1')
        self.is_running = True
        print('2')
        self.update_plot()
        print('3')
        self.update_recurrence_plot()
        print('4')
        self.change_frequency()

    def stop(self):
        self.is_running = False

def lorenz(xyz, *, s=10, r=28, b=2.667):
    x, y, z = xyz
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return np.array([x_dot, y_dot, z_dot])

class LivePlotAppLorenz:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Data and Recurrence Plot")

        # Create a figure and two subplots (one for the Lorenz attractor, one for the recurrence plot)
        self.fig = plt.Figure(figsize=(10, 5))
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax2 = self.fig.add_subplot(122)

        # Embedding the Matplotlib figure into Tkinter canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Add Start and Stop buttons to control the live plotting
        self.btn_start = ttk.Button(root, text="Start", command=self.start)
        self.btn_start.pack(side=tk.LEFT)

        self.btn_stop = ttk.Button(root, text="Stop", command=self.stop)
        self.btn_stop.pack(side=tk.RIGHT)

        # Variables to control the live plotting
        self.is_running = False
        self.xyzs = np.array([[0., 1., 1.05]]) # initialize xyzs with an initial value
        self.time_step = 0
        self.dt = 0.01

    def update_plot(self):
        if not self.is_running:
            return
        print('hello')
        new_point = self.xyzs[-1] + lorenz(self.xyzs[-1]) * self.dt
        print(new_point)
        print(self.xyzs)
        self.xyzs = np.append(self.xyzs, [new_point], axis=0)
        self.ax1.clear()
        self.ax1.plot(*self.xyzs.T, lw=0.5)
        self.ax1.set_title("Lorenz Attractor")
        self.ax1.set_xlabel("X Axis")
        self.ax1.set_ylabel("Y Axis")
        self.ax1.set_zlabel("Z Axis")

        if len(self.xyzs) > 1:
            self.update_recurrence_plot()

        self.canvas.draw()
        self.root.after(100, self.update_plot)

    def update_recurrence_plot(self):
        self.ax2.clear()

        m = 10
        T = 3

        num_vectors = np.shape(self.xyzs)[0]
        D = squareform(pdist(self.xyzs, metric='euclidean'))

        # set epsilon to 10% of max phase space diameter
        D_max = np.max(D)
        epsilon = 0.1 * D_max

        # Find points of recurrence
        hit = np.argwhere(D < epsilon)

        # Extract x and y coordinates of points of recurrence
        x_rec, y_rec = hit[:, 0], hit[:, 1]

        self.ax2.scatter(x_rec, y_rec, s=1)
        self.ax2.set_title("Recurrence Plot")
        self.ax2.set_xlabel("Vector Index")
        self.ax2.set_ylabel("Vector Index")

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.update_plot()
        self.update_recurrence_plot()
        self.change_frequency()

    def stop(self):
        self.is_running = False

if __name__ == "__main__":
    root = tk.Tk()
    app = LivePlotAppLorenz(root)
    root.mainloop()
