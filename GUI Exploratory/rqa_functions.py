from itertools import groupby
import numpy as np
import matplotlib.pyplot as plt
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.computation import RQAComputation
from pyrqa.metric import EuclideanMetric
from pyrqa.neighbourhood import FixedRadius


def calculate_rqa_measures_pyrqa(vectors, m, T, epsilon):
    time_series = TimeSeries(vectors[:, 0], embedding_dimension=m, time_delay=T)
    settings = Settings(
        time_series,
        neighbourhood=FixedRadius(epsilon),
        similarity_measure=EuclideanMetric(),
        theiler_corrector=1
    )

    computation = RQAComputation.create(settings)
    result = computation.run()

    rqa_measures = {
        "RR": round(result.recurrence_rate, 3),
        "DET": round(result.determinism, 3),
        "L": round(result.average_diagonal_line, 3),
        "Lmax": round(result.longest_diagonal_line, 3),
        "DIV": round(result.divergence, 3),
        "ENTR": round(result.entropy_diagonal_lines, 3),
        "LAM": round(result.laminarity, 3),
        "TT": round(result.trapping_time, 3)
    }

    return rqa_measures

def calculate_manual_det_lam_lmax(recurrence_matrix):
    def calculate_det(recurrence_matrix):
        diagonals = [np.diagonal(recurrence_matrix, offset=i) for i in range(-recurrence_matrix.shape[0] + 1, recurrence_matrix.shape[1])]
        diag_lengths = [len(list(group)) for diag in diagonals for value, group in groupby(diag) if value == 1]
        diag_lengths = [length for length in diag_lengths if length >= 2]
        if len(diag_lengths) == 0:
            return 0
        return sum(diag_lengths) / np.sum(recurrence_matrix)

    def calculate_lam(recurrence_matrix):
        verticals = recurrence_matrix.T
        vert_lengths = [len(list(group)) for col in verticals for value, group in groupby(col) if value == 1]
        vert_lengths = [length for length in vert_lengths if length >= 2]
        if len(vert_lengths) == 0:
            return 0
        return sum(vert_lengths) / np.sum(recurrence_matrix)

    def calculate_lmax(recurrence_matrix):
        diagonals = [np.diagonal(recurrence_matrix, offset=i) for i in range(-recurrence_matrix.shape[0] + 1, recurrence_matrix.shape[1]) if i != 0]
        diag_lengths = [len(list(group)) for diag in diagonals for value, group in groupby(diag) if value == 1]
        diag_lengths = [length for length in diag_lengths if length >= 2]
        if len(diag_lengths) == 0:
            return 0
        return max(diag_lengths)

    det2 = calculate_det(recurrence_matrix)
    lam2 = calculate_lam(recurrence_matrix)
    lmax2 = calculate_lmax(recurrence_matrix)

    return det2, lam2, lmax2


def extract_diagonal_lengths(recurrence_matrix):
    num_points = recurrence_matrix.shape[0]
    diagonals = [np.diag(recurrence_matrix, k) for k in range(-num_points + 1, num_points)]
    diag_lengths = [len(list(group)) for diag in diagonals for k, group in groupby(diag) if k == 1]
    diag_lengths = [length for length in diag_lengths if length >= 2]  # Exclude diagonals of length < 2
    return diag_lengths