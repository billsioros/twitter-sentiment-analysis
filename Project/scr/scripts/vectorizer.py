
import sys
import os
import re

import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

import numpy

from dictionary import Dictioanry

class Vectorizer:

    vector_size = 200

    bowargs = {
        "max_features": vector_size,
        "stop_words" : 'english',
        "max_df" : 0.5,
        "min_df" : 0.01
    }

    tfidfargs = {
        "max_df" : 1.0,
        "min_df" : 1,
        "max_features" : vector_size,
        "stop_words" : 'english'
    }

    w2vargs = {
        "size" : vector_size,
        "window" : 5,
        "min_count" : 2,
        "sg" : 1,
        "hs" : 0,
        "negative" : 10,
        "workers" : 2,
        "seed" : 34
    }


    def __init__(self, method='word2vec'):

        self.method = re.sub(r'''_|-|\ ''', '', method)

        if self.method == 'word2vec':
            self.underlying = Word2Vec(**self.w2vargs)
        elif self.method == 'bagofwords':
            self.underlying = CountVectorizer(**self.bowargs)
        elif self.method == 'tfidf':
            self.underlying = TfidfVectorizer(**self.tfidfargs)
        else:
            raise ValueError("'" + self.method + "' is not supported")


    def vectorize(self, preprocessor, dictionary_root='..\\..\\lexica', save=True):

        filename = '_'.join([preprocessor.filename, self.method]) + '.pkl'

        if os.path.isfile(filename):

            with open(filename, 'rb') as file:
                vectors = pickle.load(file)

                print('<LOG>: Loaded', len(vectors), 'vectors from', filename, '[' + str(len(vectors[0])), 'features each]', file=sys.stderr)

                return vectors

        return self.process(preprocessor, Dictioanry(dictionary_root), filename if save else None)


    def process(self, preprocessor, dictionary, filename):

        tweets = []
        for label in preprocessor.tweets.keys():
            tweets += preprocessor.tweets[label]

        valences = [[0.0] * len(dictionary.fullpaths)] * len(tweets)

        for i, tweet in enumerate(tweets):
            for token in tweet:
                for j in range(len(dictionary.fullpaths)):
                    if token in dictionary.valences:
                        valences[i][j] += dictionary.valences[token][j] / len(tweet)

        if self.method == 'word2vec':

            self.underlying.build_vocab(tweets)

            self.underlying.train(sentences=tweets, total_examples=len(tweets), epochs=20)

            vectors = []

            for i, tweet in enumerate(tweets):
                vector = []

                for token in tweet:
                    if token in self.underlying.wv:
                        vector.append(self.underlying.wv[token])
                    else:
                        vector.append(2.0 * numpy.random.randn(self.vector_size) - 1.0)

                vector  = numpy.mean(vector, axis=0)
                valence = numpy.asarray(valences[i])

                vectors.append(numpy.concatenate((vector, valence)))

        else:

            tweets = [' '.join(tweet) for tweet in tweets]

            vectors = [None] * len(tweets)

            for i, vector in enumerate(self.underlying.fit_transform(tweets).toarray()):
                vectors[i] = numpy.concatenate((vector, numpy.asarray(valences[i])))

        if filename:
            with open(filename, 'wb') as file:

                print('<LOG>: Saving', len(vectors), 'vectors to', filename, '[' + str(len(vectors[0])), 'features each]', file=sys.stderr)

                pickle.dump(vectors, file)

        return vectors

