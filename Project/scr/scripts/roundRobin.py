
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

def roundRobin(labels,labeledVector,unknownVector):

    positiveKeys = [key for key in labeledVector.keys() if labels[key] == 'positive']
    negativeKeys = [key for key in labeledVector.keys() if labels[key] == 'negative']
    neutralKeys = [key for key in labeledVector.keys() if labels[key] == 'neutral']

    totalTrainSet = []
    for key in labeledVector.keys():
        totalTrainSet.append(labeledVector[key])
    
    totalTestSet = []
    for key in unknownVector.keys():
        totalTestSet.append(unknownVector[key])

    #positive-negative
    knn0 = KNeighborsClassifier(n_neighbors=1)

    trainKeys0 = positiveKeys + negativeKeys
    iris_X0 = []
    iris_Y0 = []

    for key in trainKeys0:
        iris_X0.append(labeledVector[key])
    
    for key in trainKeys0:
        iris_Y0.append(labels[key])

    knn0.fit(iris_X0,iris_Y0)
    
    knn0TrainPrediction = knn0.predict_proba(totalTrainSet)
    knn0TestPrediction = knn0.predict_proba(totalTestSet)


    #positive-neutral
    knn1 = KNeighborsClassifier(n_neighbors=1)

    trainKeys1 = negativeKeys + neutralKeys
    iris_X1 = []
    iris_Y1 = []

    for key in trainKeys1:
        iris_X1.append(labeledVector[key])
    
    for key in trainKeys1:
        iris_Y1.append(labels[key])

    knn1.fit(iris_X1,iris_Y1)

    knn1TrainPrediction = knn1.predict_proba(totalTrainSet)
    knn1TestPrediction = knn1.predict_proba(totalTestSet)


   #negative-neutral
    knn2 = KNeighborsClassifier(n_neighbors=1)

    trainKeys2 = positiveKeys + negativeKeys
    iris_X2 = []
    iris_Y2 = []

    for key in trainKeys2:
        iris_X2.append(labeledVector[key])
    
    for key in trainKeys2:
        iris_Y2.append(labels[key])

    knn2.fit(iris_X2,iris_Y2)
    
    knn2TrainPrediction = knn2.predict_proba(totalTrainSet)
    knn2TestPrediction = knn2.predict_proba(totalTestSet)

    print(len(labeledVector.keys()))

   #Final Prediction

    finalTrainSet = []
    for i in range(len(labeledVector.keys())):
        finalTrainSet.append(np.concatenate((knn0TrainPrediction[i] , knn1TrainPrediction[i] , knn2TrainPrediction[i])))
    print(finalTrainSet)
    finaltotalTestSet = []
    for i in range(len(unknownVector.keys())):
        finaltotalTestSet.append(np.concatenate((knn0TestPrediction[i] , knn1TestPrediction[i] , knn2TestPrediction[i])))
    print(finaltotalTestSet)


    knn = KNeighborsClassifier(n_neighbors=1)
    
    iris_X = finalTrainSet
  
    iris_Y = []
    for key in labeledVector.keys():
        iris_Y.append(labels[key])

    knn.fit(iris_X,iris_Y)

    knnPrediction = knn.predict(finaltotalTestSet)

