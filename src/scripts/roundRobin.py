
import numpy as np
from itertools import combinations 
from sklearn.neighbors import KNeighborsClassifier

class RoundRobin:

    def __init__(self,labels,labeledVector,unknownVector):

        self.comb = combinations(['positive','negative','neutral'], 2) 
        
        self.labels = labels

        self.totalTrainSet = labeledVector
        self.totalTestSet = unknownVector

    def classify(self):
        finalTestSet = []
        finalTrainSet = []
        for combination in self.comb:
            prediction = self.RR_knn(combination,self.labels,self.totalTrainSet,self.totalTestSet, subProblem = True)

            if len(finalTrainSet) == 0:
                finalTrainSet = prediction[0]
                finalTestSet = prediction[1]
            else:
                finalTrainSet = self.appendPrediction(finalTrainSet,prediction[0])
                finalTestSet = self.appendPrediction(finalTestSet,prediction[1])
        
        finalPrediction = self.RR_knn(['positive','negative','neutral'],self.labels,finalTrainSet,finalTestSet, subProblem = False)
        
        return finalPrediction

    def RR_knn(self,combination,labels,totalTrainSet,totalTestSet, subProblem = False):

        iris_X = []
        iris_Y = []

        for label in labels:
            if label in combination:
                iris_X.append(totalTrainSet[labels.index(label)])
                iris_Y.append(label)

        knn = KNeighborsClassifier(n_neighbors=100)

        knn.fit(iris_X,iris_Y) 

        if subProblem == True:
            prediction = [knn.predict_proba(totalTrainSet),knn.predict_proba(totalTestSet)]
        else:
            prediction = knn.predict(totalTestSet)
        
        return prediction   

    def appendPrediction(self,set, prediction):
        newSet = []
        for i in range(len(set)):
            newSet.append(np.concatenate([set[i],prediction[i]]))
        return newSet
