
import os

import pickle

from util import platform

class Dictioanry:

    filename = os.path.join(os.path.curdir, 'out', 'dictionary.pkl')

    @staticmethod
    def convert(value, range_src, range_dst):
        min_src, max_src = range_src
        min_dst, max_dst = range_dst

        return min_dst + (((value - min_src) * (max_dst - min_dst)) / (max_src - min_src))


    def __init__(self, root, vector_range=None, duplicate_weight=0.5, save=False):

        if os.path.isfile(self.filename):
            with open(self.filename, mode='rb') as file:
                print('<LOG>: Loading word valences from', self.filename)

                self.fullpaths, self.valences = pickle.load(file)

                return

        if duplicate_weight < 0.0 or duplicate_weight > 1.0:
            raise ValueError("'duplicate_weight' must be a value in the range [0.0, 1.0]")

        self.fullpaths = []

        for directory, _, filenames in os.walk(platform.path(root)):
            for filename in filenames:
                self.fullpaths.append(os.path.join(root, directory, filename))

        self.valences = {}

        tmin, tmax = [float('+inf')] * len(self.fullpaths), [float('-inf')] * len(self.fullpaths)

        for index, fullpath in enumerate(self.fullpaths):

            vmin = float('+inf')
            vmax = float('-inf')

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

                    vmin = min(vmin, valences[word])
                    vmax = max(vmax, valences[word])

            for word, valence in valences.items():
                if word not in self.valences:
                    self.valences[word] = [0.0] * len(self.fullpaths)

                self.valences[word][index] = self.convert(valence, (vmin, vmax), vector_range) if vector_range else valence

                tmin[index] = min(tmin[index], self.valences[word][index])
                tmax[index] = max(tmax[index], self.valences[word][index])

        for i in range(len(self.fullpaths)):
            print('<LOG> ', self.fullpaths[i], '{ min:', tmin[i], 'max:', tmax[i], '}')

        if save:
            if not os.path.isdir('out'):
                os.mkdir('out')

            with open(self.filename, mode='wb') as file:
                print('<LOG>: Saving word valences to', self.filename)

                pickle.dump((self.fullpaths, self.valences), file)

    def per_tweet(self, tweets):

        valences = [[0.0] * len(self.fullpaths)] * len(tweets)

        for i, tweet in enumerate(tweets):
            for token in tweet:
                for j in range(len(self.fullpaths)):
                    if token in self.valences:
                        valences[i][j] += self.valences[token][j] / len(tweet)

        vmin, vmax = float('+inf'), float('-inf')

        for j in range(len(self.fullpaths)):
            vmin = min(vmin, min(valences[:][j]))
            vmax = max(vmax, max(valences[:][j]))

        print('{ min:', vmin, 'max:', vmax, '}')

        return valences


if __name__ == "__main__":

    d = Dictioanry('..\\..\\lexica')

    print("The valence of 'happy' across different dictionaries")

    for fullpath, valence in zip(d.fullpaths, d.valences['happy']):
        print(fullpath, ':', valence)

