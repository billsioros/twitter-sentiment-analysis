
from cruncher import Cruncher
from preprocessor import Preprocessor
from visualizer import Visualizer
from vectorizer import Vectorizer

if __name__ == "__main__":

    preprocessor = Preprocessor('../test.tsv', Cruncher())

    # visualization = Visualizer(preprocessor).visualize()

    for method in ['word-2-vec']:
        vectors = [vector for vector in Vectorizer(method).vectorize(preprocessor, save=False)]

        vmin = float('+inf')
        vmax = float('-inf')
        
        for vector in vectors:
            vmin = min(vmin, min(vector))
            vmax = max(vmax, max(vector))

        print('Method:', method, 'min:', vmin, 'max:', vmax)

