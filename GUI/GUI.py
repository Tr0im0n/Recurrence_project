import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from itertools import groupby
from scipy.spatial.distance import pdist, squareform
from my_functions import lorenz, chua

class LivePlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Data and Recurrence Plot")

        # Configure grid layout
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # Create a figure and three subplots
        self.fig_ps = plt.Figure()
        self.fig_comp = plt.Figure()
        self.fig_rp = plt.Figure()

        # Embedding the Matplotlib figures into Tkinter canvas
        self.canvas_ps = FigureCanvasTkAgg(self.fig_ps, master=root)
        self.canvas_ps.get_tk_widget().grid(row=0, column=1, sticky='nsew')

        self.canvas_comp = FigureCanvasTkAgg(self.fig_comp, master=root)
        self.canvas_comp.get_tk_widget().grid(row=1, column=0, sticky='nsew')

        self.canvas_rp = FigureCanvasTkAgg(self.fig_rp, master=root)
        self.canvas_rp.get_tk_widget().grid(row=1, column=1, sticky='nsew')

        # Make frame for buttons
        self.command_window = ttk.Frame(root)
        self.command_window.grid(row=0, column=0, sticky='nsew')

        self.command_window.columnconfigure(0, weight=1)
        self.command_window.columnconfigure(1, weight=1)
        self.command_window.columnconfigure(2, weight=1)
        self.command_window.rowconfigure(0, weight=1)
        self.command_window.rowconfigure(1, weight=1)
        self.command_window.rowconfigure(2, weight=1)
        self.command_window.rowconfigure(3, weight=1)

        # Add Start, Stop, and reset buttons to control the live plotting
        self.btn_start = ttk.Button(self.command_window, text="Start", command=self.start)
        self.btn_start.grid(row=0, column=0)

        self.btn_stop = ttk.Button(self.command_window, text="Stop", command=self.stop)
        self.btn_stop.grid(row=1, column=0)

        self.btn_reset = ttk.Button(self.command_window, text="Reset", command=self.reset)
        self.btn_reset.grid(row=2, column=0)

        # Add button to open histogram window
        self.btn_histogram = ttk.Button(self.command_window, text="Show Histogram", command=self.show_histogram)
        self.btn_histogram.grid(row=3, column=0)

        # Add drop down menu to select function
        self.selected_option = tk.StringVar()
        functions = ["Lorenz", "Chua"]
        self.selected_option.set(functions[0])
        self.dropdown = ttk.OptionMenu(self.command_window, self.selected_option, *functions)
        self.dropdown.grid(row=0, column=1)
        self.selected_option.trace("w", self.on_select)

        # Label to display RQA measures
        self.rqa_label = ttk.Label(self.command_window, text="RQA Measures will appear here")
        self.rqa_label.grid(row=0, column=2)

        # Variables to control the live plotting
        self.is_running = False
        self.xyzs = np.array([[0., 1., 1.05]])  # Initialize xyzs with an initial value
        self.time_step = 0
        self.dt = 0.01

    # def button_state(self, state):
    #     # Toggle all buttons except start and stop
    #     if self.is_running:
    #         self.active_command = 'disabled'
    #     else:
    #         self.active_command = 'normal'
    def toggle_buttons(self, state):
        # Toggle all buttons except start and stop
        self.btn_reset.config(state=state)
        self.btn_histogram.config(state=state)
        self.dropdown.config(state=state)

    def update_plot(self):
        if not self.is_running:
            return
        selected_function_name = self.selected_option.get().lower()
        if selected_function_name == 'lorenz':
            new_point = self.xyzs[-1] + lorenz(self.xyzs[-1]) * self.dt
        elif selected_function_name == 'chua':
            new_point = self.xyzs[-1] + chua(self.xyzs[-1]) * self.dt
        else:
            new_point = self.xyzs[-1]  # Default, no change

        self.xyzs = np.append(self.xyzs, [new_point], axis=0)
        self.fig_ps.clear()
        ax_ps = self.fig_ps.add_subplot(111, projection='3d')
        ax_ps.plot(*self.xyzs.T, lw=0.5)
        ax_ps.set_title(f"{selected_function_name.capitalize()} Attractor")
        ax_ps.set_xlabel("X Axis")
        ax_ps.set_ylabel("Y Axis")
        ax_ps.set_zlabel("Z Axis")

        self.x = [coord[0] for coord in self.xyzs]
        self.fig_comp.clear()
        ax_comp = self.fig_comp.add_subplot(111)
        ax_comp.plot(self.x, lw=0.5)
        ax_comp.set_title("X-coordinate")
        ax_comp.set_xlabel("Time")
        ax_comp.set_ylabel("X-coordinate")

        if len(self.xyzs) > 0:
            self.update_recurrence_plot()

        self.canvas_ps.draw()
        self.canvas_comp.draw()
        self.canvas_rp.draw()
        self.root.after(1, self.update_plot)

    def update_recurrence_plot(self):
        self.fig_rp.clear()
        ax_rp = self.fig_rp.add_subplot(111)

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
            ax_rp.scatter(x_rec, y_rec, s=1)
            ax_rp.set_title("Recurrence Plot")
            ax_rp.set_xlabel("Vector Index")
            ax_rp.set_ylabel("Vector Index")
            rqa_measures, diag_lengths = self.calculate_rqa_measures(recurrence_matrix)
            self.display_rqa_measures(rqa_measures)
            self.diag_lengths = diag_lengths

    def calculate_rqa_measures(self, recurrence_matrix):
        num_points = recurrence_matrix.shape[0]
        # Calculate recurrence rate (RR)
        RR = np.sum(recurrence_matrix) / (num_points ** 2)

        # Calculate diagonal line structures
        diagonals = [np.diag(recurrence_matrix, k) for k in range(-num_points + 1, num_points)]
        diag_lengths = [len(list(group)) for diag in diagonals for k, group in groupby(diag) if k == 1]

        # Calculate DET
        DET = sum(l for l in diag_lengths if l >= 2) / np.sum(recurrence_matrix) if np.sum(recurrence_matrix) != 0 else 0

        # Calculate L
        L = np.mean([l for l in diag_lengths if l >= 2]) if diag_lengths else 0

        # Calculate Lmax
        Lmax = max(diag_lengths) if diag_lengths else 0

        # Calculate DIV
        DIV = 1 / Lmax if Lmax != 0 else 0

        # Calculate ENTR
        counts = np.bincount(diag_lengths)
        probs = counts / np.sum(counts)
        probs = probs[probs > 0]
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

    def on_select(self, *args):
        selected_value = self.selected_option.get()
        print(f"Selected: {selected_value}")

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.toggle_buttons('disabled')
        self.update_plot()
        self.update_recurrence_plot()


    def stop(self):
        self.is_running = False
        self.toggle_buttons('normal')

    def reset(self):
        self.is_running = False
        self.xyzs = np.array([[0., 1., 1.05]])  # Reset the initial value
        self.fig_ps.clear()
        self.fig_comp.clear()
        self.fig_rp.clear()
        self.canvas_ps.draw()
        self.canvas_comp.draw()
        self.canvas_rp.draw()
        self.selected_option.set("Lorenz")  # Reset to default option


if __name__ == "__main__":
    root = tk.Tk()
    app = LivePlotApp(root)
    root.mainloop()
