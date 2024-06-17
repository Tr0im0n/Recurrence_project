from itertools import groupby
import numpy as np
import matplotlib.pyplot as plt
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.computation import RQAComputation
from pyrqa.metric import EuclideanMetric
from pyrqa.neighbourhood import FixedRadius

def calculate_rqa_measures_pyrqa(vectors, epsilon):
    time_series = TimeSeries(vectors[:, 0], embedding_dimension=1, time_delay=1)
    settings = Settings(
        time_series,
        neighbourhood=FixedRadius(epsilon),
        similarity_measure=EuclideanMetric(),
        theiler_corrector=1
    )

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

def calculate_manual_det_lam_lmax(recurrence_matrix):
    def calculate_det(recurrence_matrix):
        # Calculate diagonal line lengths
        diagonals = [np.diagonal(recurrence_matrix, offset=i) for i in range(-recurrence_matrix.shape[0] + 1, recurrence_matrix.shape[1])]
        diag_lengths = [len(list(group)) for diag in diagonals for value, group in groupby(diag) if value == 1]
        diag_lengths = [length for length in diag_lengths if length >= 2]
        if len(diag_lengths) == 0:
            return 0
        return sum(diag_lengths) / np.sum(recurrence_matrix)

    def calculate_lam(recurrence_matrix):
        # Calculate vertical line lengths
        verticals = recurrence_matrix.T
        vert_lengths = [len(list(group)) for col in verticals for value, group in groupby(col) if value == 1]
        vert_lengths = [length for length in vert_lengths if length >= 2]
        if len(vert_lengths) == 0:
            return 0
        return sum(vert_lengths) / np.sum(recurrence_matrix)

    def calculate_lmax(recurrence_matrix):
        diagonals = [np.diagonal(recurrence_matrix, offset=i) for i in range(-recurrence_matrix.shape[0] + 1, recurrence_matrix.shape[1])]
        diag_lengths = [len(list(group)) for diag in diagonals for value, group in groupby(diag) if value == 1]
        diag_lengths = [length for length in diag_lengths if length >= 2]
        if len(diag_lengths) == 0:
            return 0
        return max(diag_lengths)

    det2 = calculate_det(recurrence_matrix)
    lam2 = calculate_lam(recurrence_matrix)
    lmax2 = calculate_lmax(recurrence_matrix)

    return det2, lam2, lmax2

def display_rqa_measures(self, rqa_label, rqa_measures):
    text = "\n".join([f"{k}: {v:.4f}" for k, v in rqa_measures.items()])
    rqa_label.config(text=text)

def show_histogram(diag_lengths):
    unique_lengths = np.unique(diag_lengths)
    counts = [diag_lengths.count(ul) for ul in unique_lengths]
    plt.figure()
    plt.bar(unique_lengths, counts)
    plt.xlabel("Diagonal Length")
    plt.ylabel("Count")
    plt.title("Histogram of Diagonal Lengths")
    plt.show()
