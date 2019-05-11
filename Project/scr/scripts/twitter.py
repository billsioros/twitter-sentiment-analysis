
from cruncher import Cruncher
from preprocessor import Preprocessor
from visualizer import Visualizer
from vectorizer import Vectorizer
from classifier import Classifier

if __name__ == "__main__":

    preprocessor1 = Preprocessor('../test.tsv', Cruncher())

    preprocessor2 = Preprocessor('../test2017.tsv', Cruncher())

    # visualization = Visualizer(preprocessor).visualize()

    for method in ['word-2-vec']:
        vectors = Vectorizer(method).vectorize(preprocessor1, augmented=True)

        vectors, labels = list(vectors.values()), list(preprocessor1.labels.values())

        classifier = Classifier(vectors, labels)

        vectors = Vectorizer(method).vectorize(preprocessor2, augmented=True)

        print(classifier.predict(list(vectors.values())))

