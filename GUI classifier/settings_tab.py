import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial.distance import pdist, squareform

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter.filedialog as fd


class settingsTab:
    def __init__(self, root, notebook):
        # access root and notebook from main file
        self.root = root
        self.notebook = notebook

        self.settings_tab = ttk.Frame(notebook)
        notebook.add(self.settings_tab, text='Settings')
        self.create_settings_tab()

    def create_settings_tab(self):
        # # Section 1: Control of Data Stream
        # frame_data_stream = tk.LabelFrame(self.disp_tab, text="Control of Data Stream", padx=10, pady=10)
        # frame_data_stream.pack(padx=10, pady=5, fill="x")
        #
        # # Frame rate
        # tk.Label(frame_data_stream, text="Frame Rate:").pack(side="left")
        # frame_rate = tk.IntVar()
        # frame_rate_entry = tk.Entry(frame_data_stream, textvariable=frame_rate)
        # frame_rate_entry.pack(side="left", padx=5)
        #
        # # Data speed
        # tk.Label(frame_data_stream, text="Data Speed:").pack(side="left")
        # data_speed = tk.IntVar()
        # data_speed_entry = tk.Entry(frame_data_stream, textvariable=data_speed)
        # data_speed_entry.pack(side="left", padx=5)

        # Section 2: RQA Measures
        frame_rqa_measures = tk.LabelFrame(self.settings_tab, text="RQA Measures", padx=10, pady=10)
        frame_rqa_measures.pack(padx=10, pady=5, fill="x")

        self.param_options = ["Recurrence Rate", "Determinism", "Average Diagonal", "Trapping Time", "Longest Diagonal", "Divergence", "Entropy", "Laminarity"]

        # Parameter 1 Dropdown
        tk.Label(frame_rqa_measures, text="Parameter 1:").pack(side="left")
        self.param1 = tk.StringVar()
        self.param1.set(self.param_options[0])
        param1_menu = ttk.Combobox(frame_rqa_measures, textvariable=self.param1, values=self.param_options)
        param1_menu.pack(side="left", padx=5)

        # Parameter 2 Dropdown
        tk.Label(frame_rqa_measures, text="Parameter 2:").pack(side="left")
        self.param2 = tk.StringVar()
        self.param2.set(self.param_options[1])
        param2_menu = ttk.Combobox(frame_rqa_measures, textvariable=self.param2, values=self.param_options)
        param2_menu.pack(side="left", padx=5)

