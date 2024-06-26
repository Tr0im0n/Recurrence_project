"""----------------------------------------------------------------------------------------------- 
1. Load data (eg. 0 - 1 000 000)
2. use first 50 000 samples for training
3. create timeseries based on healthy -> fault (selectable)
4. simulate live data acquisition -> processing -> classification
5. plot per window: timeseries, recurrence plot, rqa graph (customizable), classification
-----------------------------------------------------------------------------------------------"""
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_svm(X_train, y_train, kernel='linear', C=1, decision_function='ovo'):
    svm_classifier = SVC(kernel=kernel, C=C, decision_function_shape=decision_function)
    svm_classifier.fit(X_train, y_train)
    return svm_classifier

def train_multiclass_classifier(X_train, y_train):
    return train_svm(X_train, y_train, decision_function='ovo')

def predict(classifier, X_test):
    return classifier.predict(X_test)

def evaluate_accuracy(y_true, predictions):
    return accuracy_score(y_true, predictions)

def load_data(filepath, column_name, num_samples):
    return pd.read_csv(filepath)[column_name][0:num_samples].values

def sliding_window_view(arr, window_size, step):
    shape = ((arr.shape[0] - window_size) // step + 1, window_size)
    strides = (step * arr.strides[0], arr.strides[0])
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

def prepare_datasets_multi_class(time_series, fault_names, window_size, delay, feature_func):
    """
    All arrays in time_series should be the same length
    """

    test1 = []  # Will become a list with arrays of rqas
    for i, series in enumerate(time_series):
        windows = sliding_window_view(series, window_size, delay)
        rqas = np.apply_along_axis(feature_func, 1, windows)
        print(f"{fault_names[i]} measures shape: {rqas.shape}")
        test1.append(rqas)

    size = test1[0].shape[0]
    amount_of_time_series = len(time_series)
    all_labels = np.zeros(amount_of_time_series*size)
    for i in range(1, amount_of_time_series):
        all_labels[i*size:(i+1)*size] = i * np.ones(size)

    # Combine the datasets
    X = np.vstack(test1)
    print("Concatenated dataset shape:", X.shape)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, all_labels, test_size=0.2)
    print("Training set shape:", X_train.shape, "Testing set shape:", X_test.shape)

    return X_train, X_test, y_train, y_test

def train_classifier(data):
    X_train, X_test, y_train, y_test = data
    classifier = train_multiclass_classifier(X_train, y_train)
    print("Classifier Accuracy: ", evaluate_accuracy(y_test, predict(classifier, X_test)))
    return classifier