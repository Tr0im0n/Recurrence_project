import pandas as pd

# Load the CSV files
file1 = pd.read_csv('/Users/martina/Documents/GitHub/Recurrence_project/Classifier/data/normal_3hp_1730rpm.csv')
file2 = pd.read_csv('/Users/martina/Documents/GitHub/Recurrence_project/Classifier/data/.007_inner_race.csv')
file3 = pd.read_csv('/Users/martina/Documents/GitHub/Recurrence_project/Classifier/data/.007_ball.csv')
file4 = pd.read_csv('/Users/martina/Documents/GitHub/Recurrence_project/Classifier/data/.007_centerd_6.csv')

#file4 = pd.read_csv('file4.csv')  # not used in this example, but loaded anyway

# Select the desired points from each file

points1 = []
sources1 = []

points2 = []
sources2 = []

points3 = []
sources3 = []

points4 = []
sources4 = []

points1.extend(file1.iloc[:, 0].tolist())
sources1.extend(['0'] * len(file1.iloc[:, 0]))

points2.extend(file2.iloc[:, 0].tolist())
sources2.extend(['1'] * len(file2.iloc[:, 0]))

points3.extend(file3.iloc[:, 0].tolist())
sources3.extend(['2'] * len(file3.iloc[:, 0]))

points4.extend(file4.iloc[:, 0].tolist())
sources4.extend(['3'] * len(file4.iloc[:, 0]))

# Create a new DataFrame with the selected points
df_healthy = pd.DataFrame({'point': points1, 'source': sources1})
df_inner = pd.DataFrame({'point': points2, 'source': sources2})
df_ball = pd.DataFrame({'point': points3, 'source': sources3})
df_outer = pd.DataFrame({'point': points4, 'source': sources4})

# Save the new CSV file
df_healthy.to_csv('healthy.csv', index=False)
df_inner.to_csv('inner.csv', index=False)
df_ball.to_csv('ball.csv', index=False)
df_outer.to_csv('outer.csv', index=False)