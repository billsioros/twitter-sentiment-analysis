
from cruncher import Cruncher
from preprocessor import Preprocessor
from visualizer import Visualizer
from vectorizer import Vectorizer

import numpy

if __name__ == "__main__":

    preprocessor = Preprocessor('../test.tsv', Cruncher())

    # visualization = Visualizer(preprocessor).visualize()

    for method in ['word-2-vec']:
        vectors = Vectorizer(method).vectorize(preprocessor)

