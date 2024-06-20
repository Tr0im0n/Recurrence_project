import numpy as np
from scipy.spatial.distance import pdist, squareform
from itertools import groupby
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# To-Do
# 1. Fix TT RQA-measure
# 2. Try differnt SVM setups
# 3. implement SVM to classify fault type
# 4. clean code 

def calc_recurrence_plots(timeseries, m, T, epsilon):
    num_vectors = len(timeseries) - (m - 1) * T
    vectors = np.array([timeseries[i:i + m*T:T] for i in range(num_vectors)])
    distance_matrix = squareform(pdist(vectors, metric='euclidean'))
    # Normalize the distance matrix
    max_distance = np.max(distance_matrix)
    if max_distance > 0:
        normalized_distance_matrix = distance_matrix / max_distance
    else:
        normalized_distance_matrix = distance_matrix

    recurrence_matrix = (normalized_distance_matrix <= epsilon).astype(int)
    return recurrence_matrix

def calc_rqa_measures(recurrence_matrix, min_line_length=2):
    time_series_length = recurrence_matrix.shape[0]
    # Calculate recurrence rate (RR)
    RR = np.sum(recurrence_matrix) / (time_series_length ** 2)

    # Calculate diagonal line structures
    diagonals = [np.diag(recurrence_matrix, k) for k in range(-time_series_length + 1, time_series_length)]
    diag_lengths = [len(list(group)) for diag in diagonals for k, group in groupby(diag) if k == 1]

    # Calculate DET
    DET = sum(l for l in diag_lengths if l >= min_line_length) / np.sum(recurrence_matrix) if np.sum(recurrence_matrix) != 0 else 0

    # Calculate L
    L = np.mean([l for l in diag_lengths if l >= min_line_length]) if diag_lengths else 0

    # Calculate Lmax
    Lmax = max(diag_lengths) if diag_lengths else 0

    # Calculate DIV
    DIV = 1 / Lmax if Lmax != 0 else 0

    # Calculate ENTR
    counts = np.bincount(diag_lengths)
    probs = counts / np.sum(counts) if np.sum(counts) > 0 else np.array([0])
    ENTR = -np.sum(probs * np.log(probs + np.finfo(float).eps)) if np.sum(counts) > 0 else 0

    # Calculate trend (TREND)
    TREND = np.mean([np.mean(recurrence_matrix[i, i:]) for i in range(len(recurrence_matrix))])

    # Calculate laminarity (LAM)
    verticals = [recurrence_matrix[:, i] for i in range(time_series_length)]
    vert_lengths = [len(list(group)) for vert in verticals for k, group in groupby(vert) if k == 1]
    LAM = sum(l for l in vert_lengths if l >= min_line_length) / np.sum(recurrence_matrix) if np.sum(recurrence_matrix) != 0 else 0

    # Calculate trapping time (TT)
    # TT = np.mean([l for l in vert_lengths if l >= min_line_length]) if vert_lengths else 0

    # Calculate maximum length of vertical structures (Vmax)
    Vmax = max(vert_lengths) if vert_lengths else 0

    # Calculate entropy of vertical structures (VENTR)
    vert_counts = np.bincount(vert_lengths)
    vert_probs = vert_counts / np.sum(vert_counts) if np.sum(vert_counts) > 0 else np.array([0])
    # VENTR = -np.sum(vert_probs * np.log(vert_probs)) if np.sum(vert_counts) > 0 else 0

    # Ratio between DET and RR
    DET_RR = DET / RR if RR > 0 else 0

    return {'RR': RR, 'DET': DET, 'LAM': LAM, 'DET_RR': DET_RR, 'L': L, 'DIV': DIV, 'ENTR': ENTR, 'TREND': TREND}

# Bearing Data Set
healthy = pd.read_csv('C:/Users/carle/OneDrive/Dokumente/GitHub/Recurrence_project/datasets/normal_3hp_1730rpm.csv')['X100_DE_time'][0:25000].values
inner_race_fault = pd.read_csv('C:/Users/carle/OneDrive/Dokumente/GitHub/Recurrence_project/datasets/InnerRace_0.028.csv')['X059_DE_time'][0:25000].values
ball_fault = pd.read_csv('C:/Users/carle/OneDrive/Dokumente/GitHub/Recurrence_project/datasets/Ball_0.028.csv')['X051_DE_time'][0:25000].values

m = 5 # embedding dimension
T = 2 # delay
epsilon = 0.1 # threshold

l = 100  # Window size
delay = 10  # Delay before calculating next RP


# Healthy measures
healthy_measures = pd.DataFrame([calc_rqa_measures(calc_recurrence_plots(healthy[window:window+l], m, T, epsilon)) for window in range(0, len(healthy) - l + 1, delay)])

# Faulty measures (combine inner race and ball fault for simplicity)
faulty_measures = pd.concat([
    pd.DataFrame([calc_rqa_measures(calc_recurrence_plots(inner_race_fault[window:window+l], m, T, epsilon)) for window in range(0, len(inner_race_fault) - l + 1, delay)]),
    pd.DataFrame([calc_rqa_measures(calc_recurrence_plots(ball_fault[window:window+l], m, T, epsilon)) for window in range(0, len(ball_fault) - l + 1, delay)])
])

# Label the data
healthy_measures['label'] = 0  # 0 for healthy
faulty_measures['label'] = 1  # 1 for faulty

# Combine the datasets
data = pd.concat([healthy_measures, faulty_measures])

# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# Split features and labels
X = data.drop('label', axis=1)
y = data['label']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0)

# Train the classifier
svm_classifier.fit(X_train, y_train)

# Make predictions
predictions = svm_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')





