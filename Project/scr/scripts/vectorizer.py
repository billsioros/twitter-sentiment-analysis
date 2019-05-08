
import os

import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

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
        "size" : 50,
        "window" : 5,
        "min_count" : 2,
        "sg" : 1,
        "hs" : 0,
        "negative" : 10,
        "workers" : 2,
        "seed" : 34
    }

    def __init__(self, preprocessor, method='word_embeddings'):

        self.preprocessor = preprocessor

        self.method = method

        if method == 'word_embeddings':
            self.underlying = Word2Vec(**self.w2vargs)

            def process(data):
                self.underlying.build_vocab(data)
                
                self.underlying.train(sentences=data, total_examples=len(data), epochs=20)

                return self.underlying

            self.process = lambda data: process(data)

            self.load = Word2Vec.load

            self.save = lambda model, filename: model.save(filename)

            self.transform = lambda data: data

            return

        elif method == 'bag_of_words':
            self.underlying = CountVectorizer(**self.bowargs)
        elif method == 'tf_idf':
            self.underlying = TfidfVectorizer(**self.tfidfargs)
        else:
            raise ValueError("'" + method + "' is not supported")

        self.process = self.underlying.fit_transform

        def load(filename):
            with open(filename, 'rb') as file:
                return pickle.load(file)

        self.load = load

        def save(data, filename):
            with open(filename, 'wb') as file:
                pickle.dump(data, file)

        self.save = save

        self.transform = lambda data: [' '.join(array) for array in data]

    def vectorize(self, labels=['positive', 'negative', 'neutral'], save=True):

        filename = '_'.join([self.preprocessor.filename, self.method] + [label for label in set(labels)]) + '.pkl'

        if os.path.isfile(filename):
            return self.load(filename)

        data = []

        for label in set(labels):
            if label not in self.preprocessor.tweets.keys():
                raise ValueError("'" + label + "' is not a valid label")
            else:
                data += self.transform(self.preprocessor.tweets[label])

        model = self.process(data)

        if save:
            self.save(model, filename)

        return model

