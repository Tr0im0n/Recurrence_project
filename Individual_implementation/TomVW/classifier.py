import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from Classifier.feature_extraction import calc_rqa_measures, pyrqa
from Individual_implementation.TomVW.pyrqaGPT import compute_rqa_measures
from Individual_implementation.TomVW.recurrence import view_cdist, threshold
# pip install openpyxl

directory_path = r"Classifier/data/"
file_names = ["normal_3hp_1730rpm.csv", "InnerRace_0.028.csv", "Ball_0.028.csv",
              ".007_inner_race.csv", ".007_ball.csv", ".007_centerd_6.csv"]

# col_names = ['X100_DE_time']  # We dont need col names?
my_fault_names = ["healthy", "Inner race fault 28", "Ball fault 28",
                  "Inner race fault 07", "Ball fault 07", "Outer race fault 07"]

rqa_names = ["RR", "DET", "L", "TT", "Lmax", "DIV", "ENTR", "LAM"]
markers = ['1', '2', '3', '4', '+', 'x']


def stride_array(array: np.ndarray, window_size: int, step: int):
    # shape = (array.shape[0] // window_size, window_size)
    shape = ((array.shape[0] - window_size) // step + 1, window_size)
    strides = (window_size * array.strides[0], array.strides[0])
    return np.lib.stride_tricks.as_strided(array, shape, strides)


def rqas_from_time_series(signal: np.ndarray, emb_dim: int = 5, time_delay: int = 2,
                          epsilon: float = 0.1) -> np.ndarray:
    d_matrix = view_cdist(signal, emb_dim, time_delay)
    max_d = np.max(d_matrix)
    r_matrix = threshold(d_matrix, epsilon*max_d)
    return calc_rqa_measures(r_matrix)


def my_pyrqa(signal: np.ndarray):
    emb_dim = 2
    time_delay = 2
    epsilon = 0.1
    # return pyrqa(signal, emb_dim, time_delay, epsilon)
    return compute_rqa_measures(signal, emb_dim, time_delay, epsilon)


def csv_to_sheet(file_name: str, i: int = 0, n_amount: int = 10_000, window_size: int = 1_000, step: int = 500):
    print("called")
    df = pd.read_csv(directory_path+file_names[i])
    time_series = df.iloc[:n_amount, 0].to_numpy()
    del df
    strided_time_series = stride_array(time_series, window_size, step)
    # rqas = np.apply_along_axis(rqas_from_time_series, 1, strided_time_series)
    # rqas = np.zeros((20, 8))
    # for i in range(20):
    #     rqas[0] = my_pyrqa(strided_time_series[i])
    rqas = np.apply_along_axis(my_pyrqa, 1, strided_time_series)
    # np.savetxt(r"Figures/data.csv", rqas, delimiter=',')
    print("done with rqas")
    df_rqas = pd.DataFrame(rqas, columns=rqa_names)
    mode = 'w' if 0 == i else 'a'
    # os.chdir(r"")
    with pd.ExcelWriter(file_name, engine='openpyxl', mode=mode) as writer:
        df_rqas.to_excel(writer, sheet_name=my_fault_names[i], index=False)


def csv_to_xlsx():
    file_name = "Figures/rqas00.xlsx"
    for i in range(6):
        csv_to_sheet(file_name, i, 100_001)
        print(i)


def xlsx_to_png():
    file_name = "Figures/rqas02.xlsx"
    dfs = [pd.read_excel(file_name, sheet_name=fault_name) for fault_name in my_fault_names]
    fig, axs = plt.subplots(2, 3)
    (ax1, ax2, ax3), (ax4, ax5, ax6) = axs
    axs_flat = ax1, ax2, ax3, ax4, ax5, ax6

    custom_range = [1, 2, 3, 5, 6, 7]
    for i, ax in zip(custom_range, axs_flat):
        # for j, df in enumerate(dfs):
        #     ax.scatter(df[rqa_names[0]], df[rqa_names[i]], s=4, label=my_fault_names[j])
        for df, fault_name, marker in zip(dfs, my_fault_names, markers):
            ax.scatter(df[rqa_names[0]], df[rqa_names[i]], label=fault_name, marker=marker)
        ax.set_xlabel(rqa_names[0])
        ax.set_ylabel(rqa_names[i])
        ax.legend()

    plt.show()


if __name__ == "__main__":
    # os.chdir(r"../..")
    csv_to_xlsx()
    # xlsx_to_png()




"""

data_path = r"Classifier/data/"
healthy_data_path = "normal_3hp_1730rpm.csv"
inner_race_fault_028_path = "InnerRace_0.028.csv"
ball_fault_028_path = "Ball_0.028.csv"
inner_race_fault_007_path = ".007_inner_race.csv"
ball_fault_007_path = ".007_ball.csv"
outer_race_fault_007_path = ".007_centerd_6.csv"


"""



