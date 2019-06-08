
import os
import sys

import pickle

from util import platform

import numpy as np

class Dictioanry:

    filename = os.path.join(os.path.curdir, 'out', 'dictionary.pkl')

    @staticmethod
    def convert(value, range_src, range_dst):
        min_src, max_src = range_src
        min_dst, max_dst = range_dst

        return min_dst + (((value - min_src) * (max_dst - min_dst)) / (max_src - min_src))


    def __init__(self, root, duplicate_weight=0.5, save=True):

        if os.path.isfile(self.filename):
            with open(self.filename, mode='rb') as file:
                print('<LOG>: Loading word valences from', self.filename, file=sys.stderr)

                self.relpaths, self.valences = pickle.load(file)

                for i in range(len(self.relpaths)):
                    elements = [values[i] for values in self.valences.values()]

                    print('<LOG>:', 'The normalized valences of', os.path.basename(self.relpaths[i]).ljust(max(map(lambda path: len(os.path.basename(path)), self.relpaths))), 'are in the range', '[' + '{0:+.4f}'.format(min(elements)), ',', '{0:+.4f}'.format(max(elements)) + ']', file=sys.stderr)

                return

        if duplicate_weight < 0.0 or duplicate_weight > 1.0:
            raise ValueError("'duplicate_weight' must be a value in the range [0.0, 1.0]")

        self.relpaths = []

        for directory, _, filenames in os.walk(platform.path(root)):
            for filename in filenames:
                self.relpaths.append(os.path.join(root, directory, filename))

        self.valences = {}

        for index, fullpath in enumerate(self.relpaths):

            valences = {}

            with open(fullpath, mode='r', encoding='ascii', errors='ignore') as file:
                for line in file.readlines():

                    line = line.strip().split()

                    words, valence = line[:-1], float(line[-1])

                    for word in words:
                        if word not in valences:
                            valences[word] = valence
                        else:
                            valences[word] = duplicate_weight * valences[word] + (1.0 - duplicate_weight) * valence

            for word, valence in valences.items():
                if word not in self.valences:
                    self.valences[word] = [0.0] * len(self.relpaths)

                self.valences[word][index] = valence

            valence_min = np.min(list(self.valences.values()))
            valence_max = np.max(list(self.valences.values()))

            print('<LOG>:', 'The valences of', os.path.basename(fullpath).ljust(max(map(lambda path: len(os.path.basename(path)), self.relpaths))), 'are in the range', '[' + '{0:+.4f}'.format(valence_min), ',', '{0:+.4f}'.format(valence_max) + ']', file=sys.stderr)

            for word in self.valences.keys():
                for index, value in enumerate(list(self.valences[word])):
                    self.valences[word][index] = self.convert(value, (valence_min, valence_max), (-1, 1))

        if save:
            if not os.path.isdir('out'):
                os.mkdir('out')

            with open(self.filename, mode='wb') as file:
                pickle.dump((self.relpaths, self.valences), file)

                print('<LOG>: Saved word valences to', self.filename, file=sys.stderr)

    def per_tweet(self, tweets, vector_range):

        valences = [[0.0] * len(self.relpaths)] * len(tweets)

        for i, tweet in enumerate(tweets):
            for j in range(len(self.relpaths)):
                valences[i][j] = np.mean([self.convert(self.valences[token][j], (-1, 1), vector_range) for token in tweet if token in self.valences])

        return valences


if __name__ == "__main__":

    dictionary = Dictioanry('..\\..\\lexica')

    print("\n<LOG>: The valence of 'happy' across different dictionaries")

    for fullpath, valence in zip(dictionary.relpaths, dictionary.valences['happy']):
        print('<LOG>:', fullpath.ljust(max(map(len, dictionary.relpaths))), ':', '{0:+.4f}'.format(valence))

