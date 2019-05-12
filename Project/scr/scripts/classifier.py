
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

class Classifier:

    def __init__(self, vectors, labels, method='svm'):

        if method == 'svm':
            self.underlying = svm.SVC(kernel='sigmoid', gamma='scale', C=1, probability=True)
        elif method == 'knn':
            self.underlying = KNeighborsClassifier(n_neighbors=100)
        else:
            raise ValueError("'" + method + "' is not supported")

        self.underlying.fit(vectors, labels)

    def predict(self, unknown):
        return self.underlying.predict(unknown)

    def predic_proba(self, unknown):
        return self.underlying.predict_proba(unknown)

