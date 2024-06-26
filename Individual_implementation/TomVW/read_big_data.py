
import pandas as pd
import os


os.chdir(r"../..")

file_name = "bearingData_3hp_1730rpmDE - Tabellenblatt1.csv"
file_path = r"Classifier/data/"
df = pd.read_csv(file_path+file_name)

print(df.describe())

