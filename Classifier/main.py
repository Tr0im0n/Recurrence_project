
import numpy as np
from preprocessing import load_data, prepare_datasets_multi_class, sliding_window_view
from feature_extraction import pyrqa
from classifier import train_multiclass_classifier, predict
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler 

""" 
To-DO:
Fix rqas with m > 4
understand effect of normalizing the timeseries.
"""
def manual_test(classifier, scaler, data, window_size, delay, feature_func2, start_sample):
    # Use data starting from start_sample
    # max = data.max()
    # data = data / max
    test_series = data[start_sample:]
    windows = sliding_window_view(test_series, window_size, delay)
    X = np.apply_along_axis(feature_func2, 1, windows)
    X_scaled = scaler.transform(X)
    test_preds = predict(classifier, X_scaled)
    return test_preds

def main():
    # Constants
    m = 20  # Embedding dimension
    T = 2  # Delay
    epsilon = 0.7  # Threshold
    l = 1000  # Window size
    delay = 200  # Delay before calculating next RP
    num_samples = 50000  # Total number of samples
    train_samples = 30000  # Number of samples to use for training


    # # Estimate the largest distance vector
    # time_series_range = np.max(timeseries) - np.min(timeseries)
    # max_distance_estimate = time_series_range * np.sqrt(m)
    # # Calculate epsilon as 10% of the estimated largest distance vector
    # epsilon = 0.1 * max_distance_estimate

    # Load data
    healthy_data_path = 'Classifier/data/normal_3hp_1730rpm.csv'
    inner_race_fault_007_path = 'Classifier/data/.007_inner_race.csv'
    ball_fault_007_path = 'Classifier/data/.007_ball.csv'
    outer_race_fault_007_path = 'Classifier/data/.007_centerd_6.csv'
    frankenstein_path = 'datasets/classefiergui.csv'

    healthy = load_data(healthy_data_path, 'X100_DE_time', num_samples)
    inner_race_fault_007 = load_data(inner_race_fault_007_path, 'X121_DE_time', num_samples)
    ball_fault_007 = load_data(ball_fault_007_path, 'X108_DE_time', num_samples)
    outer_race_fault_007 = load_data(outer_race_fault_007_path, 'X133_DE_time', num_samples)

    data = [healthy, inner_race_fault_007, ball_fault_007, outer_race_fault_007]
    fault_names = ['Healthy', 'Inner race fault', 'Ball fault', 'Outer race fault']

    # Feature extraction
    feature_func2 = lambda data: pyrqa(data, m, T, epsilon)

    # Prepare datasets
    X_train, y_train = prepare_datasets_multi_class(data, l, delay, feature_func2, train_samples)

    # Scale features with fitted scaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train classifier
    classifier = train_multiclass_classifier(X_train_scaled, y_train)

    # Save classifier and scaler
    # dump(classifier, 'classifier.joblib')
    # dump(scaler, 'scaler.joblib')

    # Manual testing on each fault type
    for i, (fault_data, fault_name) in enumerate(zip(data, fault_names)):
        test_preds = manual_test(classifier, scaler, fault_data, l, delay, feature_func2, train_samples)
        accuracy = np.mean(test_preds == i)
        print(f"Accuracy on {fault_name}: {accuracy:.4f}")

    # Test on Frankenstein dataset
    # classifier = load('GUI classifier/classifier.joblib')
    frankendata = load_data(frankenstein_path, 'point', 243000)
    # frankendata_max = frankendata.max()
    # frankendata = frankendata / frankendata_max
    franken_windows = sliding_window_view(frankendata, l, delay)
    franken_features = np.apply_along_axis(feature_func2, 1, franken_windows)
    franken_features_scaled = scaler.transform(franken_features)
    franken_preds = predict(classifier, franken_features_scaled)
    print("Frankenstein predictions:", *franken_preds, sep= '')
    
if __name__ == "__main__":
    main()
