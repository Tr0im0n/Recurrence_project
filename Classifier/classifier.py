
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_svm(X_train, y_train, kernel='linear', C=30, decision_function='ovo'):
    svm_classifier = SVC(kernel=kernel, C=C, decision_function_shape=decision_function)
    svm_classifier.fit(X_train, y_train)
    return svm_classifier

def train_binary_classifier(X_train, y_train):
    return train_svm(X_train, y_train, decision_function='ovr')

def train_multiclass_classifier(X_train, y_train):
    return train_svm(X_train, y_train, decision_function='ovo')

def predict(classifier, X_test):
    return classifier.predict(X_test)

def evaluate_accuracy(y_true, predictions):
    return accuracy_score(y_true, predictions)
