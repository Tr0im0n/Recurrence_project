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

        self.disp_tab = ttk.Frame(notebook)
        notebook.add(self.disp_tab, text='Settings')
        self.create_disp_tab()

    def create_disp_tab(self):
        # Section 1: Control of Data Stream
        frame_data_stream = tk.LabelFrame(self.disp_tab, text="Control of Data Stream", padx=10, pady=10)
        frame_data_stream.pack(padx=10, pady=5, fill="x")

        # Frame rate
        tk.Label(frame_data_stream, text="Frame Rate:").pack(side="left")
        frame_rate = tk.IntVar()
        frame_rate_entry = tk.Entry(frame_data_stream, textvariable=frame_rate)
        frame_rate_entry.pack(side="left", padx=5)

        # Data speed
        tk.Label(frame_data_stream, text="Data Speed:").pack(side="left")
        data_speed = tk.IntVar()
        data_speed_entry = tk.Entry(frame_data_stream, textvariable=data_speed)
        data_speed_entry.pack(side="left", padx=5)

        # Section 2: RQA Measures
        frame_rqa_measures = tk.LabelFrame(self.disp_tab, text="RQA Measures", padx=10, pady=10)
        frame_rqa_measures.pack(padx=10, pady=5, fill="x")

        # Parameter 1 Dropdown
        tk.Label(frame_rqa_measures, text="Parameter 1:").pack(side="left")
        param1 = tk.StringVar()
        param1.set("RR")
        param1_options = ["RR", "DET", "L", "TT", "Lmax", "DIV", "ENTR", "LAM"]
        param1_menu = ttk.Combobox(frame_rqa_measures, textvariable=param1, values=param1_options)
        param1_menu.pack(side="left", padx=5)

        # Parameter 2 Dropdown
        tk.Label(frame_rqa_measures, text="Parameter 2:").pack(side="left")
        param2 = tk.StringVar()
        param2.set("RR")
        param2_options = ["RR", "DET", "L", "TT", "Lmax", "DIV", "ENTR", "LAM"]
        param2_menu = ttk.Combobox(frame_rqa_measures, textvariable=param2, values=param2_options)
        param2_menu.pack(side="left", padx=5)

        # Section 3: Fault Types
        frame_fault_types = tk.LabelFrame(self.disp_tab, text="Fault Types", padx=10, pady=10)
        frame_fault_types.pack(padx=10, pady=5, fill="x")

        # Checkboxes for fault types
        inner_race_fault = tk.BooleanVar()
        tk.Checkbutton(frame_fault_types, text="Inner race fault", variable=inner_race_fault).pack(side="left")
        ball_fault = tk.BooleanVar()
        tk.Checkbutton(frame_fault_types, text="Ball fault", variable=ball_fault).pack(side="left")
        outer_race_fault = tk.BooleanVar()
        tk.Checkbutton(frame_fault_types, text="Outer race fault", variable=outer_race_fault).pack(side="left")