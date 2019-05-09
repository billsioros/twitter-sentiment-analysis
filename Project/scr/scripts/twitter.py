
from cruncher import Cruncher
from preprocessor import Preprocessor
from visualizer import Visualizer
from vectorizer import Vectorizer

if __name__ == "__main__":

    preprocessor = Preprocessor('../test.tsv', Cruncher())

    visualization = Visualizer(preprocessor).visualize()

    model = Vectorizer(preprocessor).vectorize()

