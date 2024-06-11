import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random

class LivePlotApp:
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
        if self.is_running:
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
        print('hello')
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
        if not self.is_running:
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


if __name__ == "__main__":
    root = tk.Tk()
    app = LivePlotApp(root)
    root.mainloop()
