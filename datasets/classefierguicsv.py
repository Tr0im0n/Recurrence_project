# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 13:04:53 2024

@author: akash
"""

import pandas as pd

# Load the CSV files
file1 = pd.read_csv('/Users/martina/Documents/GitHub/Recurrence_project/Classifier/data/normal_3hp_1730rpm.csv')
file2 = pd.read_csv('/Users/martina/Documents/GitHub/Recurrence_project/Classifier/data/.007_ball.csv')
file3 = pd.read_csv('/Users/martina/Documents/GitHub/Recurrence_project/Classifier/data/.007_inner_race.csv')
file4 = pd.read_csv('/Users/martina/Documents/GitHub/Recurrence_project/Classifier/data/.007_centerd_6.csv')

#file4 = pd.read_csv('file4.csv')  # not used in this example, but loaded anyway

# Select the desired points from each file

points = []
sources = []

points.extend(file1.iloc[50000:100000, 0].tolist())
sources.extend(['0'] * len(file1.iloc[50000:100000, 0]))

points.extend(file2.iloc[55000:100000, 0].tolist())
sources.extend(['1'] * len(file2.iloc[55000:100000, 0]))

points.extend(file1.iloc[100000:200000, 0].tolist())
sources.extend(['0'] * len(file1.iloc[100000:200000, 0]))

points.extend(file3.iloc[52000:100000, 0].tolist())
sources.extend(['2'] * len(file3.iloc[52000:100000, 0]))

# Create a new DataFrame with the selected points
df = pd.DataFrame({'point': points, 'source': sources})

# Save the new CSV file
df.to_csv('classefiergui.csv', index=False)