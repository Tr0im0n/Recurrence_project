import numpy as np
import pandas as pd
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.metric import EuclideanMetric
from pyrqa.neighbourhood import FixedRadius
from pyrqa.analysis_type import Classic
from pyrqa.computation import RQAComputation


def compute_rqa_measures(time_series: np.ndarray, embedding_dimension: int = 2,
                         time_delay: int = 1, neighbourhood_radius: float = 1.0):
    """
    Computes RQA measures for a given time series using pyrqa.

    Parameters:
    - time_series (list or np.ndarray): The input time series data.
    - embedding_dimension (int): Embedding dimension for phase space reconstruction.
    - time_delay (int): Time delay for phase space reconstruction.
    - neighbourhood_radius (float): Radius for neighbourhood.

    Returns:
    - rqa_measures (dict): Dictionary containing the specified RQA measures.
    """
    # Convert the time series into a pyrqa TimeSeries object
    ts = TimeSeries(time_series, embedding_dimension=embedding_dimension, time_delay=time_delay)

    # Set the settings for RQA
    settings = Settings(ts,
                        analysis_type=Classic,
                        neighbourhood=FixedRadius(neighbourhood_radius),
                        similarity_measure=EuclideanMetric)

    # Perform the RQA computation
    computation = RQAComputation.create(settings, verbose=True)
    result = computation.run()

    # Extract the specified RQA measures
    rqa_measures_array = np.array([
        result.recurrence_rate,
        result.determinism,
        result.average_diagonal_line,
        result.trapping_time,
        result.longest_diagonal_line,
        result.divergence,
        result.entropy_diagonal_lines,
        result.laminarity
    ])

    return rqa_measures_array


def test1():
    # Example usage
    time_series_data = np.random.randn(10_000)  # Replace with your actual time series data
    rqa_measures = compute_rqa_measures(time_series_data)
    print(rqa_measures)


if __name__ == "__main__":
    test1()

