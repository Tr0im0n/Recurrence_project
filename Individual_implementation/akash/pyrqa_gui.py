import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation


# Define the Lorenz system of differential equations
def lorenz(t, xyz, sigma=10, rho=28, beta=8 / 3):
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


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

        # Label to display RQA measures
        self.rqa_label = ttk.Label(root, text="RQA Measures will appear here")
        self.rqa_label.pack(side=tk.BOTTOM, fill=tk.X, expand=1)

        # Variables to control the live plotting
        self.is_running = False
        self.xyz = np.array([1.0, 1.0, 1.0])  # Initial conditions [x0, y0, z0]
        self.time_step = 0
        self.dt = 0.01
        self.time_series = None
        self.result = None

    def update_plot(self):
        if not self.is_running:
            return

        t = self.time_step * self.dt
        new_point = self.xyz + lorenz(t, self.xyz) * self.dt
        self.xyz = new_point

        # Update the time series with the x-coordinate of the Lorenz attractor
        if self.time_series is None:
            self.time_series = TimeSeries([self.xyz[0]], embedding_dimension=3, time_delay=1)
        else:
            self.time_series.append(self.xyz[0])

        self.ax1.clear()
        self.ax1.plot(*self.xyz.T, lw=0.5)
        self.ax1.set_title("Lorenz Attractor")
        self.ax1.set_xlabel("X Axis")
        self.ax1.set_ylabel("Y Axis")
        self.ax1.set_zlabel("Z Axis")

        self.x = self.time_series.data
        self.ax2.clear()
        self.ax2.plot(self.x, lw=0.5)
        self.ax2.set_title("X-coordinate of Lorenz Attractor")
        self.ax2.set_xlabel("Time")
        self.ax2.set_ylabel("X-coordinate")

        if self.time_series.length > 1:
            self.update_recurrence_plot()

        self.canvas.draw()
        self.root.after(100, self.update_plot)

    def update_recurrence_plot(self):
        self.ax3.clear()

        if self.result is None:
            self.calculate_rqa()

        recurrence_matrix = self.result.recurrence_matrix_reverse
        x_rec, y_rec = np.argwhere(recurrence_matrix).T
        self.ax3.scatter(x_rec, y_rec, s=1)
        self.ax3.set_title("Recurrence Plot")
        self.ax3.set_xlabel("Vector Index")
        self.ax3.set_ylabel("Vector Index")

        self.display_rqa_measures(self.result)

    def calculate_rqa(self):
        # Perform Recurrence Quantification Analysis (RQA)
        settings = Settings(self.time_series,
                            analysis_type=Classic,
                            neighbourhood=FixedRadius(0.1),
                            similarity_measure=EuclideanMetric,
                            theiler_corrector=1,
                            min_diagonal_line_length=2,
                            min_vertical_line_length=2,
                            min_white_vertical_line_length=2)
        computation = RQAComputation.create(settings, verbose=True)
        self.result = computation.run()

    def display_rqa_measures(self, result):
        rqa_measures = {
            "RR": result.recurrence_rate,
            "DET": result.determinism,
            "L": result.average_diagonal_line,
            "Lmax": result.longest_diagonal_line,
            "DIV": result.divergence,
            "ENTR": result.entropy_diagonal_lines,
            "TREND": result.trend,
            "LAM": result.laminarity,
            "TT": result.trapping_time,
            "Vmax": result.longest_vertical_line,
            "VENTR": result.entropy_vertical_lines,
            "W": result.average_white_vertical_line,
            "Wmax": result.longest_white_vertical_line,
            "WENTR": result.entropy_white_vertical_lines
        }

        text = "\n".join([f"{k}: {v:.4f}" for k, v in rqa_measures.items()])
        self.rqa_label.config(text=text)

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.update_plot()

    def stop(self):
        self.is_running = False


if __name__ == "__main__":
    root = tk.Tk()
    app = LivePlotAppLorenz(root)
    root.mainloop()
