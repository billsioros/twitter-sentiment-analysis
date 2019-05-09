
import sys
import os
import re

import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

import numpy

class Vectorizer:

    bowargs = {
        "max_features": 200,
        "stop_words" : 'english',
        "max_df" : 0.5,
        "min_df" : 0.01
    }

    tfidfargs = {
        "max_df" : 1.0,
        "min_df" : 1,
        "max_features" : 1000,
        "stop_words" : 'english'
    }

    w2vargs = {
        "size" : 100,
        "window" : 5,
        "min_count" : 2,
        "sg" : 1,
        "hs" : 0,
        "negative" : 10,
        "workers" : 2,
        "seed" : 34
    }


    def __init__(self, preprocessor, method='word_2_vec'):

        self.preprocessor = preprocessor

        self.method = re.sub(r'''-|\ ''', '_', method)

        if self.method == 'word_2_vec':
            self.underlying = Word2Vec(**self.w2vargs)
        elif self.method == 'bag_of_words':
            self.underlying = CountVectorizer(**self.bowargs)
        elif self.method == 'tf_idf':
            self.underlying = TfidfVectorizer(**self.tfidfargs)
        else:
            raise ValueError("'" + self.method + "' is not supported")


    def vectorize(self, save=True):

        filename = '_'.join([self.preprocessor.filename, self.method]) + '.pkl'

        if os.path.isfile(filename):
            print('<LOG>: Loading vectors from', filename, file=sys.stderr)

            with open(filename, 'rb') as file:
                return pickle.load(file)

        return self.process(filename if save else None)


    def process(self, filename):
        
        tweets = self.preprocessor.tweets

        if self.method == 'word_2_vec':

            self.underlying.build_vocab(tweets)

            self.underlying.train(sentences=tweets, total_examples=len(tweets), epochs=20)

            vectors = []

            for tweet in tweets:
                vector = []

                for token in tweet:
                    if token in self.underlying.wv:
                        vector.append(self.underlying.wv[token])
                    else:
                        vector.append(2 * numpy.random.randn(100) - 1)

                vectors.append(numpy.mean(vector))
                
        else:

            tweets = [' '.join(tweet) for tweet in tweets]

            vectors = self.underlying.fit_transform(tweets).toarray()

        if filename:
            with open(filename, 'wb') as file:
                print('<LOG>: Saving vectors to', filename, file=sys.stderr)

                pickle.dump(vectors, file)

        return vectors

