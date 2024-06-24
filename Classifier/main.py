
import numpy as np
from preprocessing import load_data, prepare_datasets, scale_features
from feature_extraction import calc_recurrence_plots, calc_rqa_measures
from classifier import train_classifier, predict, evaluate_accuracy

def main():
    # Constants
    m = 10 # Embedding dimension
    T = 2  # Delay
    epsilon = 0.1  # Threshold, % of largest distance vec
    l = 1000  # Window size
    delay = 100  # Delay before calculating next RP
    num_samples = 75000  # Number of samples to load

    # Load data
    healthy_data_path = 'Classifier/data/normal_3hp_1730rpm.csv'
    inner_race_fault_path = 'Classifier/data/InnerRace_0.028.csv'
    ball_fault_path = 'Classifier/data/Ball_0.028.csv'
    
    healthy = load_data(healthy_data_path, 'X100_DE_time', num_samples)
    inner_race_fault = load_data(inner_race_fault_path, 'X059_DE_time', num_samples)
    ball_fault = load_data(ball_fault_path, 'X051_DE_time', num_samples)

    # Feature extraction
    feature_func = lambda data: calc_rqa_measures(calc_recurrence_plots(data, m, T, epsilon, use_fnn=False))

    # Prepare datasets
    faulty = [inner_race_fault, ball_fault]
    X_train, X_test, y_train, y_test = prepare_datasets(healthy, faulty, l, delay, feature_func)

    # Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Train classifier
    classifier = train_classifier(X_train_scaled, y_train)

    # Predict and evaluate
    predictions = predict(classifier, X_test_scaled)
    accuracy = evaluate_accuracy(y_test, predictions)

    print(f'Accuracy: {accuracy}')

if __name__ == "__main__":
    main()
