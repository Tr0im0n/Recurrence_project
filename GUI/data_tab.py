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

        # initialize variables
        self.is_running = False
        self.data = np.array([])

        # create data tab
        self.data_tab = ttk.Frame(notebook)
        notebook.add(self.data_tab, text='Plotting Data')
        self.create_data_tab()

    def embedding_param_layout(self):
        self.embedding_param_frame = ttk.Frame(self.inputs_frame)
        self.embedding_param_frame.pack(side='left', padx=(0,10), anchor='c')

        self.embedding_param_frame.columnconfigure(0,weight=1)
        self.embedding_param_frame.columnconfigure(1, weight=3)
        self.embedding_param_frame.columnconfigure(2, weight=1)

        # Inputs for embedding parameters
        self.embedding_dim_var = tk.IntVar(value=1)
        self.time_delay_var = tk.IntVar(value=1)
        self.threshold_var = tk.DoubleVar(value=0.1)
        self.threshold_check_var = tk.BooleanVar(value=True)

        self.embedding_param_label = ttk.Label(self.embedding_param_frame, text='Embedding Parameters')
        self.embedding_param_label.grid(row=0, column=0, columnspan=3)
        self.embedding_dim_label = ttk.Label(self.embedding_param_frame, text='Embedding Dimension (m):')
        self.embedding_dim_label.grid(row=1, column=0, padx=(10, 0), pady=(10))
        self.time_delay_label = ttk.Label(self.embedding_param_frame, text='Time Delay (T):')
        self.time_delay_label.grid(row=2, column=0, padx=(10, 0))
        self.threshold_label = ttk.Label(self.embedding_param_frame, text='Threshold (e):')
        self.threshold_label.grid(row=3, column=0, padx=(10, 0), pady=10)

        self.embedding_dim_input = ttk.Entry(self.embedding_param_frame, textvariable=self.embedding_dim_var, width=5)
        self.embedding_dim_input.grid(row=1, column=1, columnspan=2, sticky='ew', padx=10, pady=(10))

        self.time_delay_input = ttk.Entry(self.embedding_param_frame, textvariable=self.time_delay_var, width=5)
        self.time_delay_input.grid(row=2, column=1, columnspan=2, sticky='ew', padx=10)

        vcmd = (self.embedding_param_frame.register(self.validate_threshold), '%P')
        self.threshold_input = ttk.Entry(self.embedding_param_frame, textvariable=self.threshold_var, width=5,
                                         validate='focusout', validatecommand=vcmd)
        self.threshold_input.grid(row=3, column=1, sticky='w', padx=(10, 0), pady=10)

        self.threshold_check_button = ttk.Checkbutton(self.embedding_param_frame, text='threshold', variable=self.threshold_check_var)
        self.threshold_check_button.grid(row=3,column=2, sticky='e', padx=(5,10), pady=10)

    def general_controls_layout(self):
        self.general_controls_frame = ttk.Frame(self.inputs_frame)
        self.general_controls_frame.pack(side='left', padx=(0,10), anchor='c')

        self.btn_load_file = ttk.Button(self.general_controls_frame, text="Load CSV", command=self.load_csv_data)
        self.btn_load_file.pack(padx=10, fill='x')

        self.btn_run_data = ttk.Button(self.general_controls_frame, text="Run", command=self.run_data)
        self.btn_run_data.pack(padx=10, fill='x')

        self.btn_reset_data = ttk.Button(self.general_controls_frame, text="Reset", command=self.reset_data)
        self.btn_reset_data.pack(padx=10, fill='x')

    def display_rqa_measures(self):
        rqa_measures = {
            "RR": 'TBD',
            "DET": 'TBD',
            "L": 'TBD',
            "Lmax": 'TBD',
            "DIV": 'TBD',
            "ENTR": 'TBD',
            "LAM": 'TBD',
            "TT": 'TBD'
        }
        # Destroy the previous frame if it exists
        self.rqa_frame = ttk.Frame(self.inputs_frame)
        self.rqa_frame.pack(side='left', padx=(0,10), anchor='c')

        self.rqa_table = tk.Frame(self.rqa_frame)
        self.rqa_table.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # create the treeview
        self.tree = ttk.Treeview(self.rqa_table, columns=("Measure", "Value"), show='headings', height=10)
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.tree.heading("Measure", text="Measure")
        self.tree.heading("Value", text="Value")

        for measure, value in rqa_measures.items():
            self.tree.insert("", "end", values=(measure, value))

        # Configure the column width
        self.tree.column("Measure", anchor=tk.CENTER, width=100)
        self.tree.column("Value", anchor=tk.CENTER, width=100)

    def update_rqa_table(self):
        # Clear existing entries in the treeview
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Insert the updated data into the treeview
        for measure, value in self.rqa_measures.items():
            self.tree.insert("", "end", values=(measure, value))

    def create_data_tab(self):
        self.inputs_frame = ttk.Frame(self.data_tab)
        self.inputs_frame.pack()

        self.display_frame = ttk.Frame(self.data_tab)
        self.display_frame.pack()

        self.general_controls_layout()
        self.embedding_param_layout()
        self.display_rqa_measures()

        # set up figures
        self.fig_data = plt.Figure()
        self.fig_rp_data = plt.Figure()

        self.canvas_comp_data = FigureCanvasTkAgg(self.fig_data, master=self.display_frame)
        self.canvas_comp_data.get_tk_widget().pack(side='left')

        self.canvas_rp_data = FigureCanvasTkAgg(self.fig_rp_data, master=self.display_frame)
        self.canvas_rp_data.get_tk_widget().pack(side='left')

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
        if self.threshold_check_var.get() == True:
            recurrence_matrix = self.D_norm < epsilon
        else:
            recurrence_matrix = self.D_norm

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
        ax_rp_data.imshow(recurrence_matrix, cmap='binary', origin='lower')
        ax_rp_data.set_title("Recurrence Plot")
        ax_rp_data.set_xlabel("Vector Index")
        ax_rp_data.set_ylabel("Vector Index")
        self.canvas_rp_data.draw()

        # calculate and display the RQA measures
        self.rqa_measures = calculate_rqa_measures_pyrqa(vectors, m, T, epsilon)
        self.update_rqa_table()

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
