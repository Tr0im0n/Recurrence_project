import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from itertools import groupby
from scipy.spatial.distance import pdist, squareform

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

        # Create a figure and three subplots
        self.fig = plt.Figure(figsize=(12, 5))
        self.ax1 = self.fig.add_subplot(131, projection='3d')
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)

        # Adjusting layout to prevent overlap
        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.5)

        # Embedding the Matplotlib figure into Tkinter canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Add Start and Stop buttons to control the live plotting
        self.btn_start = ttk.Button(root, text="Start", command=self.start)
        self.btn_start.pack(side=tk.LEFT)

        self.btn_stop = ttk.Button(root, text="Stop", command=self.stop)
        self.btn_stop.pack(side=tk.RIGHT)

        # Add button to open histogram window
        self.btn_histogram = ttk.Button(root, text="Show Histogram", command=self.show_histogram)
        self.btn_histogram.pack(side=tk.LEFT)

        # Label to display RQA measures
        self.rqa_label = ttk.Label(root, text="RQA Measures will appear here")
        self.rqa_label.pack(side=tk.BOTTOM, fill=tk.X, expand=1)

        # Variables to control the live plotting
        self.is_running = False
        self.xyzs = np.array([[0., 1., 1.05]])  # Initialize xyzs with an initial value
        self.time_step = 0
        self.dt = 0.01

    def update_plot(self):
        if not self.is_running:
            return
        new_point = self.xyzs[-1] + lorenz(self.xyzs[-1]) * self.dt
        self.xyzs = np.append(self.xyzs, [new_point], axis=0)
        self.ax1.clear()
        self.ax1.plot(*self.xyzs.T, lw=0.5)
        self.ax1.set_title("Lorenz Attractor")
        self.ax1.set_xlabel("X Axis")
        self.ax1.set_ylabel("Y Axis")
        self.ax1.set_zlabel("Z Axis")

        self.x = [coord[0] for coord in self.xyzs]
        self.ax2.clear()
        self.ax2.plot(self.x, lw=0.5)
        self.ax2.set_title("X-coordinate of Lorenz Attractor")
        self.ax2.set_xlabel("Time")
        self.ax2.set_ylabel("X-coordinate")

        if len(self.xyzs) > 0:
            self.update_recurrence_plot()

        self.canvas.draw()
        self.root.after(100, self.update_plot)

    def update_recurrence_plot(self):
        self.ax3.clear()

        m = 10
        T = 3
        num_vectors = len(self.x) - (m - 1) * T
        vectors = np.array([self.x[t:t + m * T:T] for t in range(num_vectors)])
        if vectors.size > 0:
            D = squareform(pdist(vectors, metric='euclidean'))
            D_max = np.max(D)
            epsilon = 0.1 * D_max
            recurrence_matrix = D < epsilon
            x_rec, y_rec = np.argwhere(recurrence_matrix).T
            self.ax3.scatter(x_rec, y_rec, s=1)
            self.ax3.set_title("Recurrence Plot")
            self.ax3.set_xlabel("Vector Index")
            self.ax3.set_ylabel("Vector Index")
            rqa_measures, diag_lengths = self.calculate_rqa_measures(recurrence_matrix)
            self.display_rqa_measures(rqa_measures)
            self.diag_lengths = diag_lengths

    def calculate_rqa_measures(self, recurrence_matrix):
        num_points = recurrence_matrix.shape[0]
        #for more details look at rqas overview document

        # Calculate recurrence rate (RR)
        RR = np.sum(recurrence_matrix) / (num_points ** 2)

        # Calculate diagonal line structures
        #not excluding main diagonal
        diagonals = [np.diag(recurrence_matrix, k) for k in range(-num_points + 1, num_points)]
        diag_lengths = [len(list(group)) for diag in diagonals for k, group in groupby(diag) if k == 1]

        # Calculate DET
        DET = sum(l for l in diag_lengths if l >= 2) / np.sum(recurrence_matrix) if np.sum(recurrence_matrix) != 0 else 0

        # Calculate L
        L = np.mean([l for l in diag_lengths if l >= 2]) if diag_lengths else 0

        # Calculate Lmax
        Lmax = max(diag_lengths) if diag_lengths else 0 #of main diag so falde

        # Calculate DIV
        DIV = 1 / Lmax if Lmax != 0 else 0

        # Calculate ENTR 
        #does not yet work
        counts = np.bincount(diag_lengths)
        probs = counts / np.sum(counts)
        ENTR = -np.sum(probs * np.log(probs)) if np.sum(counts) > 0 else 0

        # Calculate trend (TREND)
        TREND = np.mean([np.mean(recurrence_matrix[i, i:]) for i in range(num_points)])

        # Calculate laminarity (LAM)
        verticals = [recurrence_matrix[:, i] for i in range(num_points)]
        vert_lengths = [len(list(group)) for vert in verticals for k, group in groupby(vert) if k == 1]
        LAM = sum(l for l in vert_lengths if l >= 2) / np.sum(recurrence_matrix) if np.sum(recurrence_matrix) != 0 else 0

        # Calculate trapping time (TT)
        TT = np.mean([l for l in vert_lengths if l >= 2]) if vert_lengths else 0

        return {
            "RR": RR,
            "DET": DET,
            "L": L,
            "Lmax": Lmax,
            "DIV": DIV,
            "ENTR": ENTR,
            "TREND": TREND,
            "LAM": LAM,
            "TT": TT
        }, diag_lengths

    def display_rqa_measures(self, rqa_measures):
        text = "\n".join([f"{k}: {v:.4f}" for k, v in rqa_measures.items()])
        self.rqa_label.config(text=text)

    def show_histogram(self):
        if not hasattr(self, 'diag_lengths') or not self.diag_lengths:
            return

        # Create a new window for the histogram
        hist_window = tk.Toplevel(self.root)
        hist_window.title("Histogram of Diagonal Lengths")

        # Create a figure for the histogram
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(self.diag_lengths, bins=20, edgecolor='black')
        ax.set_title("Histogram of Diagonal Lengths")
        ax.set_xlabel("Diagonal Length")
        ax.set_ylabel("Frequency")

        # Embed the figure in the new window
        canvas = FigureCanvasTkAgg(fig, master=hist_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        canvas.draw()

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.update_plot()
        self.update_recurrence_plot()

    def stop(self):
        self.is_running = False

if __name__ == "__main__":
    root = tk.Tk()
    app = LivePlotAppLorenz(root)
    root.mainloop()
