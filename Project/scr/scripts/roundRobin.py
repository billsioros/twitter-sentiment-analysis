from sklearn.neighbors import KNeighborsClassifier

def roundRobin(tweetsDictionary,labeledVector,unknownVector):

    #positive-negative
    knn0 = KNeighborsClassifier(n_neighbors=1)
    iris_X = labeledVector[:len(tweetsDictionary['positive'])+len(tweetsDictionary['negative'])]

    iris_Y = [label for label in tweetsDictionary.keys() 
                    for _ in range(len(tweetsDictionary[label]))
                        if label in ['positive','negative']]

    knn0.fit(iris_X,iris_Y)

    knn0Prediction = knn0.predict(unknownVector)


    #positive-neutral
    knn1 = KNeighborsClassifier(n_neighbors=1)

    iris_X = labeledVector[:len(tweetsDictionary['positive'])] + labeledVector[len(tweetsDictionary['positive'])+len(tweetsDictionary['negative']):]
    
    iris_Y = [label for label in tweetsDictionary.keys() 
                    for _ in range(len(tweetsDictionary[label]))
                        if label in ['positive','neutral']]    

    knn1.fit(iris_X,iris_Y)

    knn1Prediction = knn1.predict(unknownVector)


    #negative-neutral
    knn2 = KNeighborsClassifier(n_neighbors=1)
    
    iris_X = labeledVector[len(tweetsDictionary['positive']):]
    
    iris_Y = [label for label in tweetsDictionary.keys() 
                    for _ in range(len(tweetsDictionary[label]))
                        if label in ['negative','neutral']]

    knn2.fit(iris_X,iris_Y)

    knn2Prediction = knn2.predict(unknownVector)