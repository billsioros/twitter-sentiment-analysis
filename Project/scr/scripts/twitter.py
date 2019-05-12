
from cruncher import Cruncher
from preprocessor import Preprocessor
from visualizer import Visualizer
from vectorizer import Vectorizer
from classifier import Classifier
from evaluator import Evaluator
from dictionary import Dictioanry

import time

def visualization(train_filename, cruncher_type='lemmatizer'):

    preprocessor = Preprocessor(train_filename, Cruncher(cruncher_type))

    return Visualizer(preprocessor)

def evaluation(train_filename, test_filename, dictionary_root='..\\..\\lexica', cruncher_type='lemmatizer', vectorizer_type='word2vec'):

    beg = time.time()

    try:

        vectorizer = Vectorizer(vectorizer_type)

        _, train_vectors = vectorizer.vectorize(train_filename, dictionary_root)

    except:

        cruncher = Cruncher(cruncher_type)

        train_preprocessor = Preprocessor(train_filename, cruncher)

        dictionary = Dictioanry(dictionary_root) if dictionary_root else None

        train_vectors = Vectorizer(vectorizer_type).vectorize(train_preprocessor, dictionary)

        train_vectors = list(train_vectors.values())


    test_preprocessor  = Preprocessor(test_filename, cruncher)

    test_vectors  = vectorizer.vectorize(test_preprocessor, dictionary)

    test_ids, test_vectors = list(test_vectors.keys()), list(test_vectors.values())

    train_labels = list(train_preprocessor.labels.values())

    evaluator = Evaluator()

    for classifing in ['knn', 'svm']:

        classifier = Classifier(train_vectors, train_labels, classifing)

        predictions = classifier.predict(list(test_vectors))

        for metric in ['f1-score', 'accuracy-score']:

            value = evaluator.evaluate(dict(zip(test_ids, predictions)), metric)

            print('<LOG>: The accuracy of', "'" + metric + "'", 'according to the', "'" + metric + "'", 'is', '{0:.6f}'.format(value))

    end = time.time()

    print('\n\nElapsed time:', '{0:.6f}'.format(end - beg), 'seconds')


if __name__ == "__main__":

    # '..\\..\\twitter_data\\train2017.tsv'

    evaluation('train.tsv', 'test.tsv', vectorizer_type='bagofwords')

