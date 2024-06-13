# Synthetic data application inspired by https://medium.com/@DataEngineeer/predictive-maintenance-harnessing-sensor-data-and-machine-learning-for-enhanced-equipment-757efe262d2f

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set the initial timestamp and number of data points
start_timestamp = datetime(2024, 1, 1)
num_data_points = 10000

# Generate synthetic sensor data
timestamps = [start_timestamp + timedelta(minutes=i) for i in range(num_data_points)]
temperature = np.random.normal(loc=25, scale=2, size=num_data_points)
vibration = np.random.normal(loc=0.05, scale=0.02, size=num_data_points)
pressure = np.random.normal(loc=10, scale=0.5, size=num_data_points)
failure = np.zeros(num_data_points)

# Introduce failures at random points
failure_indices = np.random.choice(range(num_data_points), size=int(num_data_points * 0.05), replace=False)
failure[failure_indices] = 1

# Create the dataframe
data = pd.DataFrame({
    'timestamp': timestamps,
    'temperature': temperature,
    'vibration': vibration,
    'pressure': pressure,
    'failure': failure
})

# Save the dataset to a CSV file
data.to_csv("sensor_data.csv", index=False)