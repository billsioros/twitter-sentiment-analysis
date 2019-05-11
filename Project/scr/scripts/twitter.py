
from cruncher import Cruncher
from preprocessor import Preprocessor
from visualizer import Visualizer
from vectorizer import Vectorizer
import roundRobin as RR

if __name__ == "__main__":

    preprocessor = Preprocessor('../train.tsv', Cruncher())

    #visualization = Visualizer(preprocessor).visualize()

    for method in ['word-2-vec']:
        knownVectors = Vectorizer(method).vectorize(preprocessor)
    
    preprocessor1 = Preprocessor('../test2017.tsv', Cruncher())

    for method in ['word-2-vec']:
        unknownVectors = Vectorizer(method).vectorize(preprocessor1)

    
    #print(knownVectors.keys())
    
    #print(preprocessor.labels)
    
    #for key in knownVectors.keys():
       # print(preprocessor.labels[key])

    #for key in unknownVectors.keys():
        #print(preprocessor1.labels[key])

    #print(preprocessor.labels)
    
    #print(unknownVectors.keys())
    #print(preprocessor1.labels)
    #print(preprocessor.by_label(['positive']))


    #print(knownVectors.keys())

    pos = [key for key in knownVectors.keys() if preprocessor.labels[key] == 'positive']
    
    #for key in pos:
        #print(knownVectors[key])

    


    RR.roundRobin(preprocessor.labels,knownVectors, unknownVectors)

