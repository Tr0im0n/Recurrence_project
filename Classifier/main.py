
import numpy as np
from preprocessing import load_data, prepare_datasets_bi_class, prepare_datasets_multi_class, scale_features
from feature_extraction import calc_recurrence_plot, calc_rqa_measures
from classifier import train_svm, train_binary_classifier, train_multiclass_classifier, predict, evaluate_accuracy

def main():
    # Constants
    m = 10 # Embedding dimension
    T = 2  # Delay
    epsilon = 0.1  # Threshold, % of largest distance vec
    l = 1000  # Window size
    delay = 100  # Delay before calculating next RP
    num_samples = 50000  # Number of samples to load
    use_multiclass = True

    # Load data
    healthy_data_path = 'Classifier/data/normal_3hp_1730rpm.csv'
    inner_race_fault_path = 'Classifier/data/InnerRace_0.028.csv'
    ball_fault_path = 'Classifier/data/Ball_0.028.csv'
    
    healthy = load_data(healthy_data_path, 'X100_DE_time', num_samples)
    race_fault = load_data(inner_race_fault_path, 'X059_DE_time', num_samples)
    ball_fault = load_data(ball_fault_path, 'X051_DE_time', num_samples)

    # Feature extraction
    feature_func = lambda data: calc_rqa_measures(calc_recurrence_plot(data, m, T, epsilon, use_fnn=False))

    if use_multiclass:
        X_train, X_test, y_train, y_test = prepare_datasets_multi_class(healthy, race_fault, ball_fault, l, delay, feature_func)
        # Scale features
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
        classifier = train_multiclass_classifier(X_train_scaled, y_train)
    else:
        # Prepare datasets
        faulty = [race_fault, ball_fault]
        X_train, X_test, y_train, y_test = prepare_datasets_bi_class(healthy, faulty, l, delay, feature_func)

        # Scale features
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
        classifier = train_binary_classifier(X_train_scaled, y_train)
        
    # Predict and evaluate
    predictions = predict(classifier, X_test_scaled)
    accuracy = evaluate_accuracy(y_test, predictions)

    print(f'Accuracy: {accuracy}')

if __name__ == "__main__":
    main()
