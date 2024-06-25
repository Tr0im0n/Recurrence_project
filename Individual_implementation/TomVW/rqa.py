
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from Classifier.feature_extraction import calc_rqa_measures, calc_recurrence_plot
from Classifier.preprocessing import load_data, sliding_window_view


m = 20   # Embedding dimension
T = 2  # Delay
epsilon = 0.1  # Threshold, % of largest distance vec
window_size = 1000  # Window size
delay = 100  # Delay before calculating next RP
num_samples = 50000

a_list = ["RR", "DET", "L", "TT", "Lmax", "DIV", "ENTR", "LAM"]
b_list = ["healthy", "inner race", "ball"]

os.chdir(r"..")
os.chdir(r"..")

print(os.getcwd())

# Load data
healthy_data_path = r'Classifier/data/normal_3hp_1730rpm.csv'
inner_race_fault_path = r'Classifier/data/InnerRace_0.028.csv'
ball_fault_path = r'Classifier/data/Ball_0.028.csv'
paths = [healthy_data_path, inner_race_fault_path, ball_fault_path]

healthy_col = 'X100_DE_time'
inner_race_col = 'X059_DE_time'
ball_col = 'X051_DE_time'
col_names = [healthy_col, inner_race_col, ball_col]

time_series = [load_data(path, col_name, num_samples) for path, col_name in zip(paths, col_names)]


def feature_func(data):
    return calc_rqa_measures(calc_recurrence_plot(data, m, T, epsilon, use_fnn=False))


test1 = []
for series in time_series:
    windows = sliding_window_view(series, window_size, delay)
    rqas = np.apply_along_axis(feature_func, 1, windows)
    test1.append(rqas)

scaler = StandardScaler()
test2 = [scaler.fit_transform(i) for i in test1]

fig, axs = plt.subplots(2, 3)
(ax1, ax2, ax3), (ax4, ax5, ax6) = axs
axs_flat = ax1, ax2, ax3, ax4, ax5, ax6

custom_range = [1, 2, 3, 5, 6, 7]
for i, ax in zip(custom_range, axs_flat):
    for j, rqa in enumerate(test1[:2]):
        ax.scatter(rqa[:, 0], rqa[:, i], s=4, label=b_list[j])
    ax.set_xlabel(a_list[0])
    ax.set_ylabel(a_list[i])
    ax.legend()


plt.show()


