from scipy.io import loadmat
import pandas as pd
import numpy as np

# Load the .mat file
file_path = r"C:\Users\akash\Downloads\DataForClassification_TimeDomain.mat"
data = loadmat(file_path)

# Filter out the metadata and keep only the data entries
data = {k: v for k, v in data.items() if not k.startswith('_')}

# Find the maximum length among all arrays
max_length = max(len(v.flatten()) for v in data.values())

# Prepare the dictionary for the DataFrame
data_for_df = {}
for k, v in data.items():
    flattened = v.flatten()
    # Pad the array with NaN values if necessary
    if len(flattened) < max_length:
        padded = np.pad(flattened, (0, max_length - len(flattened)), constant_values=np.nan)
    else:
        padded = flattened
    data_for_df[k] = padded

# Convert the data to a Pandas DataFrame
df = pd.DataFrame(data_for_df)

# Save the DataFrame to a CSV file
output_csv_path = "gearvibrations.csv"
df.to_csv(output_csv_path, index=False)

print(f"Data has been successfully saved to {output_csv_path}")
