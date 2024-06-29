import pandas as pd
import numpy as np
import math


def make_window(path, time, window_size):
    csv_data = pd.read_csv(path)
    # data = np.array(csv_data['X100_FE_time'])
    # data = np.array(csv_data['point'])
    # labels = np.array(csv_data['source'])
    data = csv_data.iloc[:,0]
    labels = csv_data.iloc[:, 1]

    data_max = np.max(data)
    norm_data = data / data_max

    sampling_rate = 12000 # in Herz (samples/second)

    # points wrap around data
    point = time * sampling_rate
    total_points = len(data)
    iterations = point // total_points
    current_index = point - total_points * iterations

    data_range_start = int(current_index - window_size)
    data_range_stop = int(current_index)

    return norm_data[data_range_start:data_range_stop], labels[data_range_start:data_range_stop]

