from itertools import groupby
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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