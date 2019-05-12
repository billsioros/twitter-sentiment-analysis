
import numpy as np
from itertools import combinations 
from sklearn.neighbors import KNeighborsClassifier

def roundRobin(labels,labeledVector,unknownVector):

    comb = combinations(['positive','negative','neutral'], 2) 

    totalTrainSet = []
    for key in labeledVector.keys():
        totalTrainSet.append(labeledVector[key])

    totalTestSet = []
    for key in unknownVector.keys():
        totalTestSet.append(unknownVector[key])
    
    finalTestSet = []
    finalTrainSet = []
    for combination in comb:
        prediction = RR_knn(combination,labeledVector,labels,totalTrainSet,totalTestSet, subProblem = True)

        if len(finalTrainSet) == 0:
            finalTrainSet = prediction[0]
            finalTestSet = prediction[1]
        else:
            finalTrainSet = appendPrediction(finalTrainSet,prediction[0])
            finalTestSet = appendPrediction(finalTestSet,prediction[1])

    finalPrediction = RR_knn(['positive','negative','neutral'],labeledVector,labels,totalTrainSet,totalTestSet, subProblem = False)

    print(finalPrediction)

def RR_knn(combination,labeledVector,labels,totalTrainSet,totalTestSet, subProblem = False):
    
    trainKeys = []
    iris_X = []
    iris_Y = []
    
    for comb in combination:
        trainKeys += [key for key in labeledVector.keys() if labels[key] == comb]
    
    for key in trainKeys:
        iris_X.append(labeledVector[key])
    
    for key in trainKeys:
        iris_Y.append(labels[key])
    
    knn = KNeighborsClassifier(n_neighbors=1)

    knn.fit(iris_X,iris_Y) 

    if subProblem == True:
        prediction = [knn.predict_proba(totalTrainSet),knn.predict_proba(totalTestSet)]
    else:
        prediction = knn.predict(totalTestSet)

    return prediction   

def appendPrediction(set, prediction):
            newSet = []
            for i in range(len(set)):
                newSet.append(np.concatenate([set[i],prediction[i]]))
            return newSet
