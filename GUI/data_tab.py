import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial.distance import pdist, squareform
from rqa_functions import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter.filedialog as fd


class dataTab:
    def __init__(self, root, notebook):
        # access root and notebook from main file
        self.root = root
        self.notebook = notebook

        # create data tab
        self.data_tab = ttk.Frame(notebook)
        notebook.add(self.data_tab, text='Plotting Data')
        self.create_data_tab()

        # initialize variables
        self.is_running = False
        self.data = np.array([])

    def create_data_tab(self):
        # set up layout
        self.data_tab.columnconfigure(0, weight=1)
        self.data_tab.columnconfigure(1, weight=1)
        self.data_tab.rowconfigure(0, weight=1)
        self.data_tab.rowconfigure(1, weight=1)

        self.command_window_data = ttk.Frame(self.data_tab)
        self.command_window_data.grid(row=0, column=0, columnspan=2, sticky='nsew')

        self.command_window_data.columnconfigure(0, weight=1)
        self.command_window_data.columnconfigure(1, weight=1)
        self.command_window_data.columnconfigure(2, weight=1)
        self.command_window_data.columnconfigure(3, weight=1)

        self.command_window_data.rowconfigure(0, weight=1)
        self.command_window_data.rowconfigure(1, weight=1)
        self.command_window_data.rowconfigure(2, weight=1)
        self.command_window_data.rowconfigure(3, weight=1)

        # set up figures
        self.fig_data = plt.Figure()
        self.fig_rp_data = plt.Figure()

        self.canvas_comp_data = FigureCanvasTkAgg(self.fig_data, master=self.data_tab)
        self.canvas_comp_data.get_tk_widget().grid(row=1, column=0, sticky='nsew')

        self.canvas_rp_data = FigureCanvasTkAgg(self.fig_rp_data, master=self.data_tab)
        self.canvas_rp_data.get_tk_widget().grid(row=1, column=1, sticky='nsew')

        # configure command window
        # general controls
        self.btn_load_file = ttk.Button(self.command_window_data, text="Load CSV", command=self.load_csv_data)
        self.btn_load_file.grid(row=0, column=0)

        self.btn_run_data = ttk.Button(self.command_window_data, text="Run", command=self.run_data)
        self.btn_run_data.grid(row=1, column=0)

        self.btn_reset_data = ttk.Button(self.command_window_data, text="Reset", command=self.reset_data)
        self.btn_reset_data.grid(row=2, column=0)

        # Inputs for embedding parameters
        self.embedding_dim_var = tk.IntVar(value=1)
        self.time_delay_var = tk.IntVar(value=1)
        self.threshold_var = tk.DoubleVar(value=0.1)

        self.embedding_dim_label = ttk.Label(self.command_window_data, text="Embedding Dimension (m):")
        self.embedding_dim_label.grid(row=0, column=1, sticky='e')
        self.embedding_dim_input = ttk.Entry(self.command_window_data, textvariable=self.embedding_dim_var)
        self.embedding_dim_input.grid(row=0, column=2, sticky='w')

        self.time_delay_label = ttk.Label(self.command_window_data, text="Time Delay (T):")
        self.time_delay_label.grid(row=1, column=1, sticky='e')
        self.time_delay_input = ttk.Entry(self.command_window_data, textvariable=self.time_delay_var)
        self.time_delay_input.grid(row=1, column=2, sticky='w')

        self.threshold_label = ttk.Label(self.command_window_data, text="Threshold (e):")
        self.threshold_label.grid(row=2, column=1, sticky='e')
        # check that threshold input value is decimal between 0 and 1
        vcmd = (self.command_window_data.register(self.validate_threshold), '%P')
        self.threshold_input = ttk.Entry(self.command_window_data, textvariable=self.threshold_var,
                                              validate='focusout', validatecommand=vcmd)
        self.threshold_input.grid(row=2, column=2, sticky='w')

        # RQA display
        self.rqa_label_data = ttk.Label(self.command_window_data, text="RQA Measures will appear here")
        self.rqa_label_data.grid(row=0, column=3, rowspan=2)

    def load_csv_data(self):
        file_path = fd.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.data_np = pd.read_csv(file_path)
            self.data = self.data_np.to_numpy().flatten()  # Flatten to 1D array for simplicity
            self.run_data()
            self.plot_data()

    def validate_threshold(self, value_if_allowed):
        if not value_if_allowed:
            self.threshold_var.set(0.1)
            return True
        try:
            value = float(value_if_allowed)
            if 0.0 <= value <= 1.0:
                return True
        except ValueError:
            pass
        self.show_error("Please enter a float between 0.0 and 1.0.")
        self.threshold_var.set(0.1)
        return False

    def show_error(self, message):
        messagebox.showerror("Input Error", message)

    def plot_data(self):
        # If no file is provided, use default file
        if self.data.size == 0:
            file_path = 'data_files/vibration_data_synthetic.csv'
            self.data_np = pd.read_csv(file_path, sep='\s+')
            self.data = self.data_np.to_numpy().reshape(-1, 1)

        # Need to adjust for what to do with large input files
        self.data_trunc = self.data[:1000]

        # Get embedding parameters from user inputs
        m = self.embedding_dim_var.get()        # integer value
        T = self.time_delay_var.get()           # integer value
        epsilon = self.threshold_var.get()      # float value (between 0 and 1)

        # embed time series
        num_vectors = len(self.data_trunc) - (m - 1) * T
        vectors = np.array([self.data_trunc[t:t + m * T:T] for t in range(num_vectors)]).reshape(-1, 1)

        # calculate similarity matrix
        self.D = squareform(pdist(vectors, metric='euclidean'))
        self.D_max = np.max(self.D)
        self.D_norm = self.D / self.D_max

        # threshold similarity matrix to produce recurrence plot
        recurrence_matrix = self.D_norm < epsilon
        x_rec, y_rec = np.argwhere(recurrence_matrix).T

        # plot the data
        self.fig_data.clear()
        ax_data = self.fig_data.add_subplot(111)
        ax_data.plot(self.data, lw=0.5)
        ax_data.set_title("Visualization of Inputted Data")
        ax_data.set_xlabel("X Axis")
        ax_data.set_ylabel("Y Axis")
        self.canvas_comp_data.draw()

        # plot the recurrence plot
        self.fig_rp_data.clear()
        ax_rp_data = self.fig_rp_data.add_subplot(111)
        ax_rp_data.scatter(x_rec, y_rec, s=1)
        ax_rp_data.set_title("Recurrence Plot")
        ax_rp_data.set_xlabel("Vector Index")
        ax_rp_data.set_ylabel("Vector Index")
        self.canvas_rp_data.draw()

        # calculate and display the RQA measures
        rqa_measures_data = calculate_rqa_measures_pyrqa(vectors, epsilon)
        det2, lam2, lmax2 = calculate_manual_det_lam_lmax(recurrence_matrix)
        rqa_measures_data["DET2"] = det2
        rqa_measures_data["LAM2"] = lam2
        rqa_measures_data["Lmax2"] = lmax2
        display_rqa_measures(self.rqa_label_data, rqa_measures_data)

    def run_data(self):
        if not self.is_running:
            self.is_running = True
        self.plot_data()

    def reset_data(self):
        self.is_running = False
        self.fig_data.clear()
        self.fig_rp_data.clear()
        self.canvas_comp_data.draw()
        self.canvas_rp_data.draw()
