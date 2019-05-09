
import os

from util import platform

class Dictioanry:

    @staticmethod
    def convert(value, range_src, range_dst=(-1, 1)):
        min_src, max_src = range_src
        min_dst, max_dst = range_dst

        return min_dst + (((value - min_src) * (max_dst - min_dst)) / (max_src - min_src))


    def __init__(self, root, duplicate_weight=0.5):

        if duplicate_weight < 0.0 or duplicate_weight > 1.0:
            raise ValueError("'duplicate_weight' must be a value in the range [0.0, 1.0]")

        self.valences = {}

        self.files = []

        for directory, _, filenames in os.walk(platform.path(root)):
            for filename in filenames:
                self.files.append(os.path.join(root, directory, filename))

        for index, file in enumerate(self.files):

            vmin = float('+inf')
            vmax = float('-inf')

            valences = {}

            with open(file, mode='r', encoding='ascii', errors='ignore') as file:
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
                    self.valences[word] = [0.0] * len(self.files)

                self.valences[word][index] = self.convert(valence, (vmin, vmax))

