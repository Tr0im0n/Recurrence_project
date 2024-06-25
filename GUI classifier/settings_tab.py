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
        notebook.add(self.disp_tab, text='Live Window')
        self.create_disp_tab()

        test_label = ttk.Label(self.disp_tab, text='hello world')
        test_label.pack()

    def create_disp_tab(self):
        pass