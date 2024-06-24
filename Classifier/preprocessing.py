
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(filepath, column_name, num_samples):
    return pd.read_csv(filepath)[column_name][0:num_samples].values

def sliding_window_view(arr, window_size, step):
    shape = ((arr.shape[0] - window_size) // step + 1, window_size)
    strides = (step * arr.strides[0], arr.strides[0])
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

def prepare_datasets_multi_class(healthy, race_fault, ball_fault, window_size, delay, feature_func):

    # Create features for healthy data
    healthy_windows = sliding_window_view(healthy, window_size, delay)
    healthy_measures = np.apply_along_axis(feature_func, 1, healthy_windows)
    print("Healthy measures shape:", healthy_measures.shape)
    
    # Create features for race fault data
    race_fault_windows = sliding_window_view(race_fault, window_size, delay)
    race_fault_measures = np.apply_along_axis(feature_func, 1, race_fault_windows)
    print("Race fault measures shape:", race_fault_measures.shape)

    # Create features for ball fault data
    ball_fault_windows = sliding_window_view(ball_fault, window_size, delay) 
    ball_fault_measures = np.apply_along_axis(feature_func, 1, ball_fault_windows)
    print("Ball fault measures shape:", ball_fault_measures.shape)
    
    size = healthy_measures.shape[0]
    # Label the data
    healthy_labels = np.zeros(size)
    race_fault_labels = np.ones(size)
    ball_fault_labels = np.ones(size) * 2
    
    print("Labels shape:", (healthy_labels.shape, race_fault_labels.shape, ball_fault_labels.shape))
    
    # Combine the datasets
    X = np.vstack((healthy_measures, race_fault_measures, ball_fault_measures))
    y = np.hstack((healthy_labels, race_fault_labels, ball_fault_labels))
    print("Concatenated dataset shape:", X.shape)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print("Training set shape:", X_train.shape, "Testing set shape:", X_test.shape)
    
    return X_train, X_test, y_train, y_test

def prepare_datasets_bi_class(healthy, faulty, window_size, delay, feature_func):
    # Create features for healthy data
    healthy_windows = sliding_window_view(healthy, window_size, delay)
    healthy_measures = np.apply_along_axis(feature_func, 1, healthy_windows)
    
    # Print healthy data feature shapes
    print(f'Healthy data feature shape: {healthy_measures.shape}')
    
    # Create features for faulty data
    faulty_windows = np.concatenate([sliding_window_view(fault, window_size, delay) for fault in faulty])
    faulty_measures = np.apply_along_axis(feature_func, 1, faulty_windows)
    
    # Print faulty data feature shapes
    print(f'Faulty data feature shape: {faulty_measures.shape}')
    
    # Balance the dataset sizes
    min_size = min(len(healthy_measures), len(faulty_measures))
    np.random.seed(42)  # for reproducibility
    healthy_indices = np.random.choice(len(healthy_measures), min_size, replace=False)
    faulty_indices = np.random.choice(len(faulty_measures), min_size, replace=False)
    
    healthy_measures = healthy_measures[healthy_indices]
    faulty_measures = faulty_measures[faulty_indices]
    
    # Label the data
    healthy_labels = np.zeros(min_size)
    faulty_labels = np.ones(min_size)
    
    # Combine the datasets
    X = np.vstack((healthy_measures, faulty_measures))
    y = np.hstack((healthy_labels, faulty_labels))
    
    # Print combined dataset shapes
    print(f'Combined dataset shape: {X.shape}')
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Print training and testing set shapes
    # print(f'Training set shape: {X_train.shape}')
    # print(f'Testing set shape: {X_test.shape}')
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
