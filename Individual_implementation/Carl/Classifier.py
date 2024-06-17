import numpy as np
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.computation import RPComputation
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric

# Step 1: Generate Synthetic Training Data
np.random.seed(42)
signal_list = []
label_list = []

for i in range(100):
    t = np.linspace(0, 1, 1000)
    frequency = np.random.uniform(1, 10)
    amplitude = np.random.uniform(1, 5)
    base_signal = amplitude * np.sin(2 * np.pi * frequency * t)
    noise = np.random.normal(0, 0.5, len(t))
    base_signal += noise

    if np.random.rand() < 0.5:
        spike_time = np.random.randint(100, 900)
        base_signal[spike_time] += np.random.uniform(10, 20)
        label_list.append("faulty")
    else:
        label_list.append("healthy")

    signal_list.append(base_signal)


plt.plot(signal_list[0])
plt.show()

# Step 2: Calculate RQA Measures
RQA_measures_list = []

for signal in signal_list:
    time_series = TimeSeries(signal, embedding_dimension=2, time_delay=1)
    settings = Settings(time_series, 
                        neighbourhood=FixedRadius(1.0), 
                        similarity_measure=EuclideanMetric(),
                        theiler_corrector=1)
    computation = RPComputation.create(settings)
    recurrence_plot = computation.run()
    RQA_measures = recurrence_plot.recurrence_matrix()
    RQA_measures_list.append([RQA_measures])  # Encapsulate in list to match feature set format

# Step 3: Train SVM Classifier
X_train, X_test, y_train, y_test = train_test_split(RQA_measures_list, label_list, test_size=0.2, random_state=42)
clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Classification Accuracy:", accuracy)
