import numpy as np
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.computation import RQAComputation
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric

# worst classifier EUW
def generate_synthetic_data(num_signals=100, seed=42):
    np.random.seed(seed)
    signals, labels = [], []
    for _ in range(num_signals):
        t = np.linspace(0, 1, 1000)
        
        # Sum of three sines
        signal = (
            np.random.uniform(1, 5) * np.sin(2 * np.pi * np.random.uniform(1, 10) * t) +
            np.random.uniform(1, 5) * np.sin(2 * np.pi * np.random.uniform(1, 10) * t) +
            np.random.uniform(1, 5) * np.sin(2 * np.pi * np.random.uniform(1, 10) * t)
        ) + np.random.normal(0, 0.5, len(t))
        
        label = "healthy"
        if np.random.rand() < 0.5:
            spike_time = np.random.randint(100, 900)
            signal[spike_time] += np.random.uniform(10, 20)  # Spike adjustment
            label = "faulty"
        signals.append(signal)
        labels.append(label)
    return signals, labels

def calculate_rqa_measures(signals, embedding_dimension=2, time_delay=1, radius=1.0):
    rqa_measures_list = []
    for signal in signals:
        time_series = TimeSeries(signal, embedding_dimension=embedding_dimension, time_delay=time_delay)
        settings = Settings(time_series, 
                            neighbourhood=FixedRadius(radius), 
                            similarity_measure=EuclideanMetric(),
                            theiler_corrector=1)
        computation = RQAComputation.create(settings)
        result = computation.run()
        rqa_measures = [
            result.recurrence_rate, 
            result.determinism,
            result.average_diagonal_line,
            result.longest_diagonal_line,
            result.entropy_diagonal_lines
        ]
        rqa_measures_list.append(rqa_measures)
    return rqa_measures_list

def train_and_evaluate_classifier(features, labels, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Main execution
signals, labels = generate_synthetic_data()
plt.plot(signals[0])
plt.title("Example Signal")
plt.show()

rqa_measures = calculate_rqa_measures(signals)
accuracy = train_and_evaluate_classifier(rqa_measures, labels)
print("Classification Accuracy:", accuracy)
