
from cruncher import Cruncher
from preprocessor import Preprocessor
from visualizer import Visualizer
from vectorizer import Vectorizer
from classifier import Classifier
from evaluator import Evaluator
from dictionary import Dictioanry

import sys
import time

def visualization(train_filename, cruncher_type='lemmatizer'):

    preprocessor = Preprocessor([train_filename], Cruncher(cruncher_type))

    return Visualizer(preprocessor)

def evaluation(filenames, dictionary_root='..\\..\\lexica', cruncher_type='lemmatizer', vectorizer_type='word2vec', metrics = ['f1-score', 'accuracy-score']):

    if not isinstance(filenames, list):
        raise ValueError("'" + filenames + "' is not an instance of 'list'")

    beg = time.time()


    vectorizer = Vectorizer(vectorizer_type)

    try:

        labels, vectors = vectorizer.vectorize(filenames, dictionary_root)

    except:

        preprocessor = Preprocessor(filenames, Cruncher(cruncher_type))

        dictionary = Dictioanry(dictionary_root) if dictionary_root else None

        labels, vectors = vectorizer.vectorize(preprocessor, dictionary)


    test_ids,  test_labels,  test_vectors  = [], [], []
    train_ids, train_labels, train_vectors = [], [], []

    for id, label in labels.items():

        if label == 'unknown':
            test_ids.append(id)
            test_labels.append(label)
            test_vectors.append(vectors[id])
            
        else:
            train_ids.append(id)
            train_labels.append(label)
            train_vectors.append(vectors[id])

    evaluator = Evaluator()

    for classifing in ['knn', 'svm']:

        classifier = Classifier(train_vectors, train_labels, classifing)

        predictions = classifier.predict(test_vectors)

        for metric in metrics:

            value = evaluator.evaluate(dict(zip(test_ids, predictions)), metric)

            print('<LOG>: The performance of', "'" + classifing + "'", 'according to the', ("'" + metric + "'").ljust(max(map(len, metrics)) + 2), "metric is", '{0:.6f}'.format(value))


    end = time.time()

    print('\n\nElapsed time:', '{0:.6f}'.format(end - beg), 'seconds', file=sys.stderr)


if __name__ == "__main__":

    evaluation(['..\\..\\twitter_data\\train2017.tsv', '..\\..\\twitter_data\\test2017.tsv'])
    # evaluation(['train.tsv', 'test.tsv'])
