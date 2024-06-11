import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def lorenz(xyz, *, s=10, r=28, b=2.667):
    x, y, z = xyz
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return np.array([x_dot, y_dot, z_dot])


class LivePlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Data and Recurrence Plot")

        self.fig = Figure(figsize=(10, 5))

        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax2 = self.fig.add_subplot(122)

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.btn_start = ttk.Button(root, text="Start", command=self.start)
        self.btn_start.pack(side=tk.LEFT)

        self.btn_stop = ttk.Button(root, text="Stop", command=self.stop)
        self.btn_stop.pack(side=tk.RIGHT)

        self.is_running = False
        self.xyzs = np.empty((0, 3))
        self.time_step = 0
        self.dt = 0.01

        self.update_plot()

    def update_plot(self):
        if self.is_running:
            self.xyzs = np.append(self.xyzs, [self.xyzs[-1] + lorenz(self.xyzs[-1]) * self.dt], axis=0)
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

        m = 5
        T = 1
        num_vectors = len(self.xyzs) - (m - 1) * T
        vectors = np.array([self.xyzs[t:t + m * T:T, 0] for t in range(num_vectors)])

        D = np.zeros((num_vectors, num_vectors))
        for i in range(num_vectors):
            for j in range(num_vectors):
                D[i, j] = np.linalg.norm(vectors[i] - vectors[j])

        epsilon = 0.1
        hit = np.argwhere(D < epsilon)
        x_rec, y_rec = hit[:, 0], hit[:, 1]

        self.ax2.scatter(x_rec, y_rec, s=1)
        self.ax2.set_title("Recurrence Plot")
        self.ax2.set_xlabel("Vector Index")
        self.ax2.set_ylabel("Vector Index")

    def start(self):
        if not self.is_running:
            self.is_running = True
            if len(self.xyzs) == 0:
                self.xyzs = np.array([[0., 1., 1.05]])
            self.update_plot()

    def stop(self):
        self.is_running = False


if __name__ == "__main__":
    root = tk.Tk()
    app = LivePlotApp(root)
    root.mainloop()
