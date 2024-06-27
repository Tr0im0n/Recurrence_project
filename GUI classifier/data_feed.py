import pandas as pd
import numpy as np
import math


def make_window(path, time, window_size):
    # data = np.genfromtxt(path, delimiter=',', skip_header=1, filling_values=np.nan)
    csv_data = pd.read_csv(path)
    # data = np.array(csv_data['X100_FE_time'])
    # data = np.array(csv_data['point'])
    # labels = np.array(csv_data['source'])
    data = csv_data.iloc[:,0]
    labels = csv_data.iloc[:, 1]


    data_max = np.max(data)
    norm_data = data / data_max

    sampling_rate = 12000 # in Herz (samples/second)
    # window_size = 1000

    data_range_start = int(time * sampling_rate - window_size)
    data_range_stop = int(time * sampling_rate)

    return norm_data[data_range_start:data_range_stop], labels[data_range_start:data_range_stop]

# def feed_data(path):
#     # data = np.genfromtxt(path, delimiter=',', skip_header=1, filling_values=np.nan)
#     csv_data = pd.read_csv(path)
#     data = np.array(csv_data['X100_FE_time'])
#
#     data_range_start = 0
#     data_range_stop = 10000
#
#     data = data[data_range_start:data_range_stop]
#
#     window_size = 1000
#     step_size = 100
#
#     windows = []
#
#     count = 0
#     while True:
#         window = data[count:count + window_size]
#         count += step_size
#
#         if len(window) < window_size:  # exit loop when reach end of data
#             break
#
#         windows.append(window)
#
#     return windows
#
# path = '/Users/martina/Documents/GitHub/Recurrence_project/Classifier/data/normal_3hp_1730rpm.csv'
# feed_data(path)
