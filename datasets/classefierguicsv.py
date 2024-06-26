# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 13:04:53 2024

@author: akash
"""

import pandas as pd

# Load the CSV files
file1 = pd.read_csv('normal_3hp_1730rpm.csv')
file2 = pd.read_csv('.007_ball.csv')
file3 = pd.read_csv('.007_inner_race.csv')
#file4 = pd.read_csv('file4.csv')  # not used in this example, but loaded anyway

# Select the desired points from each file

points = []
sources = []

points.extend(file1.iloc[50000:60000, 0].tolist())
points.extend(file2.iloc[55000:59000, 0].tolist())
points.extend(file1.iloc[62000:80000, 0].tolist())
points.extend(file3.iloc[52000:72000, 0].tolist())

sources.extend(['1'] * 10000)
sources.extend(['2'] * 4000)
sources.extend(['1'] * 18000)
sources.extend(['3'] * 20000)

# Create a new DataFrame with the selected points
df = pd.DataFrame({'point': points, 'source': sources})

# Save the new CSV file
df.to_csv('classefiergui.csv', index=False)