from itertools import groupby
import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.computation import RQAComputation
from pyrqa.metric import EuclideanMetric
from pyrqa.neighbourhood import FixedRadius


def calculate_rqa_measures_pyrqa(self, vectors, epsilon):
    time_series = TimeSeries(vectors[:, 0], embedding_dimension=10, time_delay=3)
    settings = Settings(time_series,
                        neighbourhood=FixedRadius(epsilon),
                        similarity_measure=EuclideanMetric(),
                        theiler_corrector=1)

    computation = RQAComputation.create(settings)
    result = computation.run()

    rqa_measures = {
        "RR": result.recurrence_rate,
        "DET": result.determinism,
        "L": result.average_diagonal_line,
        "Lmax": result.longest_diagonal_line,
        "DIV": result.divergence,
        "ENTR": result.entropy_diagonal_lines,
        "LAM": result.laminarity,
        "TT": result.trapping_time
    }

    return rqa_measures


def calculate_manual_det_lam(self, recurrence_matrix):
    # Calculate DET2
    diagonals = [np.diagonal(recurrence_matrix, offset=i) for i in
                 range(-recurrence_matrix.shape[0] + 1, recurrence_matrix.shape[1])]
    det2 = sum([np.sum(diag) for diag in diagonals if len(diag) > 1]) / np.sum(recurrence_matrix)

    # Calculate LAM2
    vertical_lines = [np.sum(recurrence_matrix[:, i]) for i in range(recurrence_matrix.shape[1])]
    lam2 = sum([length for length in vertical_lines if length > 1]) / np.sum(recurrence_matrix)

    return det2, lam2


def display_rqa_measures(self, rqa_label, rqa_measures):
    text = "\n".join([f"{k}: {v:.4f}" for k, v in rqa_measures.items()])
    print(text)
    rqa_label.config(text=text)

def show_histogram(self):
    if hasattr(self, 'diag_lengths'):
        diag_lengths = self.diag_lengths
        unique_lengths = np.unique(diag_lengths)
        counts = [diag_lengths.count(ul) for ul in unique_lengths]
        plt.figure()
        plt.bar(unique_lengths, counts)
        plt.xlabel("Diagonal Length")
        plt.ylabel("Count")
        plt.title("Histogram of Diagonal Lengths")
        plt.show()