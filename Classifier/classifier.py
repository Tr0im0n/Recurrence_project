
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_classifier(X_train, y_train):
    svm_classifier = SVC(kernel='linear', C=0.7) 
    svm_classifier.fit(X_train, y_train)
    return svm_classifier

def predict(classifier, X_test):
    return classifier.predict(X_test)

def evaluate_accuracy(y_true, predictions):
    return accuracy_score(y_true, predictions)
