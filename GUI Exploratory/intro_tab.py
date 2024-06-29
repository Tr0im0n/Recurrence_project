import tkinter as tk
import tkinter.filedialog as fd
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial.distance import pdist, squareform
import importlib.util
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np

from attractor_functions import *
from rqa_functions import *


class introTab:
    def __init__(self, root, notebook):
        # access root and notebook from main file
        self.root = root
        self.notebook = notebook

        self.rqa_measures = {
            "RR": 'TBD',
            "DET": 'TBD',
            "L": 'TBD',
            "Lmax": 'TBD',
            "DIV": 'TBD',
            "ENTR": 'TBD',
            "LAM": 'TBD',
            "TT": 'TBD'
        }

        # create function tab
        self.intro_tab = tk.Frame(notebook)
        notebook.add(self.intro_tab, text='Intro to RPs')
        self.create_intro_tab()

    def create_intro_tab(self):
        self.command_frame = ttk.Frame(self.intro_tab)
        self.command_frame.place(relx=0, rely=0, relwidth=0.5, relheight=1)

        self.display_frame = ttk.Frame(self.intro_tab)
        self.display_frame.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)

        self.embedding_param_frame = ttk.Frame(self.command_frame)
        self.embedding_param_frame.pack()

        self.sine_frame = ttk.Frame(self.command_frame)
        self.sine_frame.pack()

        self.noise_frame = ttk.Frame(self.command_frame)
        self.noise_frame.pack()

        self.spike_frame = ttk.Frame(self.command_frame)
        self.spike_frame.pack()

        self.create_param_frame()
        self.create_sine_frame()
        self.create_display_frame()
        self.plot_function()
        self.plot_rp()

    def create_param_frame(self):
        # Creating Tkinter variables
        self.embedding_dim_var = tk.IntVar(value=15)  # Default value as midpoint
        self.time_delay_var = tk.IntVar(value=5)  # Default value as midpoint
        self.threshold_var = tk.DoubleVar(value=0.5)  # Default value as midpoint

        # Embedding Dimensions Scale
        self.embedding_scale = ttk.Scale(self.embedding_param_frame, from_=0, to=30, orient='horizontal', variable=self.embedding_dim_var,
                                         command=self.update_value)
        self.embedding_scale.grid(row=0, column=1, padx=10, pady=10)
        ttk.Label(self.embedding_param_frame, text="Embedding Dimensions:").grid(row=0, column=0)

        # Time Delay Scale
        self.time_delay_scale = ttk.Scale(self.embedding_param_frame, from_=0, to=10, orient='horizontal', variable=self.time_delay_var,
                                          command=self.update_value)
        self.time_delay_scale.grid(row=1, column=1, padx=10, pady=10)
        ttk.Label(self.embedding_param_frame, text="Time Delay:").grid(row=1, column=0)

        # Threshold Scale
        self.threshold_scale = ttk.Scale(self.embedding_param_frame, from_=0.0, to=1.0, orient='horizontal', variable=self.threshold_var,
                                         command=self.update_value)
        self.threshold_scale.grid(row=2, column=1, padx=10, pady=10)
        ttk.Label(self.embedding_param_frame, text="Threshold:").grid(row=2, column=0)

        # This will update the label showing the current value of the scales
        self.value_label = ttk.Label(self.embedding_param_frame, text="")
        self.value_label.grid(row=3, column=0, columnspan=2)

    def update_value(self, event=None):
        current_values = f"Embedding: {self.embedding_dim_var.get()}, Delay: {self.time_delay_var.get()}, Threshold: {self.threshold_var.get():.2f}"
        self.value_label.configure(text=current_values)

    def create_sine_frame(self):
        self.sin1_active = tk.BooleanVar(value=True)
        self.sin2_active = tk.BooleanVar(value=False)
        self.sin3_active = tk.BooleanVar(value=False)

        self.amplitude_var1 = tk.DoubleVar(value=1)
        self.frequency_var1 = tk.DoubleVar(value=1)
        self.phase_var1 = tk.DoubleVar(value=0)
        self.vertical_var1 = tk.DoubleVar(value=0)


        row_frame1 = ttk.Frame(self.sine_frame, padding="5 5 5 5")
        row_frame1.grid(row=1, column=0, sticky=(tk.W, tk.E))

        active_entry = ttk.Checkbutton(row_frame1, variable=self.sin1_active)
        active_entry.pack(side='left')

        amplitude_entry = ttk.Entry(row_frame1, width=5, textvariable=self.amplitude_var1)
        amplitude_entry.pack(side='left')

        ttk.Label(row_frame1, text="sin(").pack(side='left')

        frequency_entry = ttk.Entry(row_frame1, width=5, textvariable=self.frequency_var1)
        frequency_entry.pack(side='left')

        ttk.Label(row_frame1, text=" * x + ").pack(side='left')

        phase_shift_entry = ttk.Entry(row_frame1, width=5, textvariable=self.phase_var1)
        phase_shift_entry.pack(side='left')

        ttk.Label(row_frame1, text=") + ").pack(side='left')

        vertical_shift_entry = ttk.Entry(row_frame1, width=5, textvariable=self.vertical_var1)
        vertical_shift_entry.pack(side='left')

        self.amplitude_var2 = tk.DoubleVar(value=1)
        self.frequency_var2 = tk.DoubleVar(value=1)
        self.phase_var2 = tk.DoubleVar(value=0)
        self.vertical_var2 = tk.DoubleVar(value=0)

        row_frame2 = ttk.Frame(self.sine_frame, padding="5 5 5 5")
        row_frame2.grid(row=2, column=0, sticky=(tk.W, tk.E))

        active_entry = ttk.Checkbutton(row_frame2, variable=self.sin2_active)
        active_entry.pack(side='left')

        amplitude_entry = ttk.Entry(row_frame2, width=5, textvariable=self.amplitude_var2)
        amplitude_entry.pack(side='left')

        ttk.Label(row_frame2, text="sin(").pack(side='left')

        frequency_entry = ttk.Entry(row_frame2, width=5, textvariable=self.frequency_var2)
        frequency_entry.pack(side='left')

        ttk.Label(row_frame2, text=" * x + ").pack(side='left')

        phase_shift_entry = ttk.Entry(row_frame2, width=5, textvariable=self.phase_var2)
        phase_shift_entry.pack(side='left')

        ttk.Label(row_frame2, text=") + ").pack(side='left')

        vertical_shift_entry = ttk.Entry(row_frame2, width=5, textvariable=self.vertical_var2)
        vertical_shift_entry.pack(side='left')

        self.amplitude_var3 = tk.DoubleVar(value=1)
        self.frequency_var3 = tk.DoubleVar(value=1)
        self.phase_var3 = tk.DoubleVar(value=0)
        self.vertical_var3 = tk.DoubleVar(value=0)

        row_frame3 = ttk.Frame(self.sine_frame, padding="5 5 5 5")
        row_frame3.grid(row=3, column=0, sticky=(tk.W, tk.E))

        active_entry = ttk.Checkbutton(row_frame3, variable=self.sin3_active)
        active_entry.pack(side='left')

        amplitude_entry = ttk.Entry(row_frame3, width=5, textvariable=self.amplitude_var3)
        amplitude_entry.pack(side='left')

        ttk.Label(row_frame3, text="sin(").pack(side='left')

        frequency_entry = ttk.Entry(row_frame3, width=5, textvariable=self.frequency_var3)
        frequency_entry.pack(side='left')

        ttk.Label(row_frame3, text=" * x + ").pack(side='left')

        phase_shift_entry = ttk.Entry(row_frame3, width=5, textvariable=self.phase_var3)
        phase_shift_entry.pack(side='left')

        ttk.Label(row_frame3, text=") + ").pack(side='left')

        vertical_shift_entry = ttk.Entry(row_frame3, width=5, textvariable=self.vertical_var3)
        vertical_shift_entry.pack(side='left')

        self.plot_button = ttk.Button(self.intro_tab, text='plot', command=self.plot_function)
        self.plot_button.pack()

    def create_display_frame(self):
        self.fig_func = plt.Figure()

        self.canvas_func = FigureCanvasTkAgg(self.fig_func, master=self.display_frame)
        self.canvas_func.get_tk_widget().pack(fill='both', expand=True)

        self.fig_rp = plt.Figure()

        self.canvas_func = FigureCanvasTkAgg(self.fig_rp, master=self.display_frame)
        self.canvas_func.get_tk_widget().pack(fill='both', expand=True)

    def plot_function(self):
        xs = np.linspace(0, 2 * np.pi, 1000)  # Adjust range to fit typical sine wave periods

        sin1 = self.amplitude_var1.get() * np.sin(self.frequency_var1.get() * xs + self.phase_var1.get()) + self.vertical_var1.get()
        sin2 = self.amplitude_var2.get() * np.sin(self.frequency_var2.get() * xs + self.phase_var2.get()) + self.vertical_var2.get()
        sin3 = self.amplitude_var3.get() * np.sin(self.frequency_var3.get() * xs + self.phase_var3.get()) + self.vertical_var3.get()

        sin = (sin1*int(self.sin1_active.get())) + (sin2*int(self.sin2_active.get())) + (sin3*int(self.sin3_active.get()))

        self.fig_func.clear()
        ax_func = self.fig_func.add_subplot(111)
        ax_func.plot(xs, sin, lw=0.5)
        ax_func.set_title("Function")
        ax_func.set_xlabel("X Axis")
        ax_func.set_ylabel("Y Axis")
        self.fig_func.tight_layout()

        self.canvas_func.draw()

    def plot_rp(self):
        pass
        # self.m = self.embedding_dim_var.get()
        # self.T = self.time_delay_var.get()
        # self.epsilon = self.threshold_var.get()
        #
        # # embed time series
        # self.num_vectors = len(data) - (self.m - 1) * self.T
        # self.vectors = np.array([data[t:t + self.m * self.T:self.T] for t in range(self.num_vectors)])
        #
        # if self.vectors.size > 0:  # check that enough points exist to create recurrence plot
        #     # create and normalize similarity matrix
        #     self.D = squareform(pdist(self.vectors, metric='euclidean'))
        #     D_max = np.max(self.D)
        #     self.D_norm = self.D / D_max
        #
        #     # create recurrence matrix
        #     if self.threshold_check_var.get():
        #         recurrence_matrix = self.D_norm < self.epsilon
        #     else:
        #         recurrence_matrix = self.D_norm
        #     return recurrence_matrix





