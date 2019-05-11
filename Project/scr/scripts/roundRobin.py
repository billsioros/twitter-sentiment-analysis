
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

def roundRobin(tweetsDictionary,labeledVector,unknownVector):

    #positive-negative
    knn0 = KNeighborsClassifier(n_neighbors=1)
    iris_X0 = labeledVector[:len(tweetsDictionary['positive'])+len(tweetsDictionary['negative'])]

    iris_Y0 = [label for label in tweetsDictionary.keys() 
                    for _ in range(len(tweetsDictionary[label]))
                        if label in ['positive','negative']]

    knn0.fit(iris_X0,iris_Y0)

    knn0TrainPrediction = knn0.predict_proba(labeledVector)
    knn0TestPrediction = knn0.predict_proba(unknownVector)


    #positive-neutral
    knn1 = KNeighborsClassifier(n_neighbors=1)

    iris_X1 = labeledVector[:len(tweetsDictionary['positive'])] + labeledVector[len(tweetsDictionary['positive'])+len(tweetsDictionary['negative']):]
    
    iris_Y1 = [label for label in tweetsDictionary.keys() 
                    for _ in range(len(tweetsDictionary[label]))
                        if label in ['positive','neutral']]    

    knn1.fit(iris_X1,iris_Y1)

    knn1TrainPrediction = knn1.predict_proba(labeledVector)
    knn1TestPrediction = knn1.predict_proba(unknownVector)


    #negative-neutral
    knn2 = KNeighborsClassifier(n_neighbors=1)
    
    iris_X2 = labeledVector[len(tweetsDictionary['positive']):]
    
    iris_Y2 = [label for label in tweetsDictionary.keys() 
                    for _ in range(len(tweetsDictionary[label]))
                        if label in ['negative','neutral']]

    knn2.fit(iris_X2,iris_Y2)

    knn2TrainPrediction = knn2.predict_proba(labeledVector)
    knn2TestPrediction = knn2.predict_proba(unknownVector) 


    #Final Prediction

    finalTrainSet = []
    for i in range(len(labeledVector)):
        finalTrainSet.append(np.concatenate((knn0TrainPrediction[i] , knn1TrainPrediction[i] , knn2TrainPrediction[i])))

    finalTestSet = []
    for i in range(len(unknownVector)):
        finalTestSet.append(np.concatenate((knn0TestPrediction[i] , knn1TestPrediction[i] , knn2TestPrediction[i])))
   
    
    knn = KNeighborsClassifier(n_neighbors=1)
    
    iris_X = finalTrainSet
    
    iris_Y = [label for label in tweetsDictionary.keys() 
                    for _ in range(len(tweetsDictionary[label]))
                        if label in ['positive','negative','neutral']]

    knn.fit(iris_X,iris_Y)

    knnPrediction = knn.predict(finalTestSet)