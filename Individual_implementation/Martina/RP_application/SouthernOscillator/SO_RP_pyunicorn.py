import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm
from pyunicorn.timeseries import RecurrenceNetwork, RecurrencePlot

# Read the data from the file
file_path = 'SO_data'
data_all = pd.read_csv(file_path, sep='\s+')

# Extract the SOI column and convert to numpy array
data = data_all['SOI'].values

# Plot the SOI data
plt.plot(data)
plt.title('SOI Data Plot')
plt.xlabel('Index')
plt.ylabel('SOI')
plt.show()

SO_rp = RecurrencePlot(data, dim=3, tau=1, threshold=4.406)

plt.matshow(SO_rp.recurrence_matrix())
plt.gca().invert_yaxis()
plt.title('Recurrence Plot')
plt.xlabel("$n$"); plt.ylabel("$n$");
plt.show()
