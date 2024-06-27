import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from Classifier.feature_extraction import calc_rqa_measures
from Individual_implementation.TomVW.recurrence import view_cdist, threshold
# pip install openpyxl

directory_path = r"Classifier/data/"
file_names = ["normal_3hp_1730rpm.csv", "InnerRace_0.028.csv", "Ball_0.028.csv",
              ".007_inner_race.csv", ".007_ball.csv", ".007_centerd_6.csv"]

# col_names = ['X100_DE_time']  # We dont need col names?
my_fault_names = ["healthy", "Inner race fault 28", "Ball fault 28",
                  "Inner race fault 07", "Ball fault 07", "Outer race fault 07"]

rqa_names = ["RR", "DET", "L", "TT", "Lmax", "DIV", "ENTR", "LAM"]


def stride_array(array: np.ndarray, window_size: int, step: int):
    # shape = (array.shape[0] // window_size, window_size)
    shape = ((array.shape[0] - window_size) // step + 1, window_size)
    strides = (window_size * array.strides[0], array.strides[0])
    return np.lib.stride_tricks.as_strided(array, shape, strides)


def rqas_from_time_series(signal: np.ndarray) -> np.ndarray:
    emb_dim = 5
    time_delay = 2
    epsilon = 0.1
    d_matrix = view_cdist(signal, emb_dim, time_delay)
    max_d = np.max(d_matrix)
    r_matrix = threshold(d_matrix, epsilon*max_d)
    return calc_rqa_measures(r_matrix)


def csv_to_csv(file_name: str, i: int = 0, n_amount: int = 10_000, window_size: int = 1_000, step: int = 500):
    print("called")
    df = pd.read_csv(directory_path+file_names[i])
    time_series = df.iloc[:n_amount, 0].to_numpy()
    del df
    strided_time_series = stride_array(time_series, window_size, step)
    rqas = np.apply_along_axis(rqas_from_time_series, 1, strided_time_series)
    # np.savetxt(r"Figures/data.csv", rqas, delimiter=',')
    print("done with rqas")
    df_rqas = pd.DataFrame(rqas, columns=rqa_names)
    mode = 'w' if 0 == i else 'a'
    # os.chdir(r"")
    with pd.ExcelWriter(file_name, engine='openpyxl', mode=mode) as writer:
        df_rqas.to_excel(writer, sheet_name=my_fault_names[i], index=False)


if __name__ == "__main__":
    os.chdir(r"../..")
    for i in range(2):
        csv_to_csv("Figures/rqas00.xlsx", i, 100_000)
        print(i)


"""

data_path = r"Classifier/data/"
healthy_data_path = "normal_3hp_1730rpm.csv"
inner_race_fault_028_path = "InnerRace_0.028.csv"
ball_fault_028_path = "Ball_0.028.csv"
inner_race_fault_007_path = ".007_inner_race.csv"
ball_fault_007_path = ".007_ball.csv"
outer_race_fault_007_path = ".007_centerd_6.csv"


"""



