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

        self.function_frame = ttk.Frame(self.command_frame)
        self.function_frame.pack()


        self.noise_frame = ttk.Frame(self.command_frame)
        self.noise_frame.pack()

        self.spike_frame = ttk.Frame(self.command_frame)
        self.spike_frame.pack()

        self.create_sine_frame()

    def create_sine_frame(self):
        self.sin1_active = tk.BooleanVar(value=True)
        self.sin2_active = tk.BooleanVar()
        self.sin2_active = tk.BooleanVar()


        # input_frame = ttk.Frame(root, padding="10 10 10 10")
        # input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Variables to store the entries for each sine function
        amplitude_entries = []
        frequency_entries = []
        phase_shift_entries = []
        vertical_shift_entries = []

        # Loop to create input fields for three sine functions
        # for i in range(3):
        #     row_frame = ttk.Frame(input_frame, padding="5 5 5 5")
        #     row_frame.grid(row=i, column=0, sticky=(tk.W, tk.E))
        #
        #
        #     # Add checkbox to activate/deactivate sine
        #     self.sin1_checkbox = ttk.Checkbutton()
        #
        #     # Add labels and entry boxes for the sine wave parameters
        #     amplitude_entry = ttk.Entry(row_frame, width=5)
        #     amplitude_entry.pack(side='left')
        #     amplitude_entries.append(amplitude_entry)
        #
        #     ttk.Label(row_frame, text="sin(").pack(side='left')
        #
        #     frequency_entry = ttk.Entry(row_frame, width=5)
        #     frequency_entry.pack(side='left')
        #     frequency_entries.append(frequency_entry)
        #
        #     ttk.Label(row_frame, text=" * x + ").pack(side='left')
        #
        #     phase_shift_entry = ttk.Entry(row_frame, width=5)
        #     phase_shift_entry.pack(side='left')
        #     phase_shift_entries.append(phase_shift_entry)
        #
        #     ttk.Label(row_frame, text=") + ").pack(side='left')
        #
        #     vertical_shift_entry = ttk.Entry(row_frame, width=5)
        #     vertical_shift_entry.pack(side='left')
        #     vertical_shift_entries.append(vertical_shift_entry)
        #
        # print(self.sin1_active.get())




