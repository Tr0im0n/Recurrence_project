
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath, column_name, num_samples):
    return pd.read_csv(filepath)[column_name][0:num_samples].values

def prepare_datasets(healthy, faulty, window_size, delay, feature_func):
    # Create features for healthy data
    healthy_measures = pd.DataFrame([
        feature_func(healthy[window:window+window_size]) for window in range(0, len(healthy) - window_size + 1, delay)
    ])
    
    # Print healthy data feature shapes
    print(f'Healthy data feature shape: {healthy_measures.shape}')
    
    # Create features for faulty data
    faulty_measures = pd.DataFrame([
        feature_func(fault[window:window+window_size]) 
        for fault in faulty
        for window in range(0, len(fault) - window_size + 1, delay)
    ])
    
    # Print faulty data feature shapes
    print(f'Faulty data feature shape: {faulty_measures.shape}')
    
    # Balance the dataset sizes
    min_size = min(len(healthy_measures), len(faulty_measures))
    healthy_measures = healthy_measures.sample(n=min_size)
    faulty_measures = faulty_measures.sample(n=min_size)
    
    # Label the data
    healthy_measures['label'] = 0
    faulty_measures['label'] = 1
    
    # Combine and shuffle the datasets
    data = pd.concat([healthy_measures, faulty_measures])
    data = data.sample(frac=1).reset_index(drop=True)
    
    # Print combined dataset shapes
    print(f'Combined dataset shape: {data.shape}')
    
    # Split data into features and labels
    X = data.drop('label', axis=1)
    y = data['label']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Print training and testing set shapes
    print(f'Training set shape: {X_train.shape}')
    print(f'Testing set shape: {X_test.shape}')
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
