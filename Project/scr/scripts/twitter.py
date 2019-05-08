
from cruncher import Cruncher
from preprocessor import Preprocessor
from visualizer import Visualizer
from vectorizer import Vectorizer

if __name__ == "__main__":

    preprocessor = Preprocessor('../test.tsv', Cruncher())

    visualizer = Visualizer(preprocessor).visualize()

    vectorizer = Vectorizer(preprocessor).vectorize()

