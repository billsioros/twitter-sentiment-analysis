
import sys
import os
import re

import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

import numpy as np

from preprocessor import Preprocessor

from util import platform

class Vectorizer:

    vector_size = 300

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


    def vectorize(self, preprocessor, dictionary, save=True):

        if isinstance(preprocessor, list):

            path = platform.filename(preprocessor, ['preprocessed', self.method] + (['augmented'] if dictionary else [])) + '.pkl'

            if not os.path.isfile(path):
                raise ValueError("'" + path + "' is not a file")

            with open(path, 'rb') as file:
                labels, vectors = pickle.load(file)

                print('<LOG>: Loaded', len(vectors), 'vectors from', path, '[' + str(len(list(vectors.values())[0])), 'features each]', file=sys.stderr)

                return dict(zip(vectors.keys(), labels)), vectors

        path = '_'.join([preprocessor.path, self.method] + (['augmented'] if dictionary else [])) + '.pkl'

        if not isinstance(preprocessor, Preprocessor):
            raise ValueError("'preprocessor' is not an instance of 'Preprocessor'")

        return self.process(preprocessor, dictionary, path if save else None)


    def process(self, preprocessor, dictionary, path):

        tweets = list(preprocessor.tweets.values())

        if self.method == 'word2vec':

            self.underlying.build_vocab(tweets)

            self.underlying.train(sentences=tweets, total_examples=len(tweets), epochs=20)

            vectors = [None] * len(tweets)

            for i, tweet in enumerate(tweets):
                vector = [None] * len(tweet)

                for j, token in enumerate(tweet):
                    if token in self.underlying.wv:
                        vector[j] = self.underlying.wv[token]
                    else:
                        vector[j] = 2.0 * np.random.randn(self.vector_size) - 1.0

                vectors[i] = np.mean(vector, axis=0)

        else:

            concatenated = [' '.join(tweet) for tweet in tweets]

            vectors = self.underlying.fit_transform(concatenated).toarray()

        if dictionary:

            flattened = list(np.asarray(vectors).flatten())

            vmin, vmax = min(flattened), max(flattened)

            augmented = [None] * len(vectors)

            for i, valences in enumerate(dictionary.per_tweet(tweets, (vmin, vmax))):
                augmented[i] = np.concatenate((vectors[i], valences))

            vectors = augmented

        print('<LOG>: The', ('augmented ' if augmented else '') + 'vectors\' values are in the range', '[' + '{0:.4f}'.format(vmin), ',', '{0:.4f}'.format(vmax) + ']', file=sys.stderr)

        vectors = dict(zip(preprocessor.tweets.keys(), vectors))

        if path:
            with open(path, 'wb') as file:

                pickle.dump((list(preprocessor.labels.values()), vectors), file)

                print('<LOG>: Saved', len(vectors), 'vectors to', path, '[' + str(len(list(vectors.values())[0])), 'features each]', file=sys.stderr)

        return preprocessor.labels, vectors

