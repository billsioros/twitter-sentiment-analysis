
from cruncher import Cruncher
from preprocessor import Preprocessor
from visualizer import Visualizer
from vectorizer import Vectorizer

if __name__ == "__main__":

   #preprocessor = Preprocessor('../../twitter_data/train2017.tsv', Cruncher())
    preprocessor = Preprocessor('test.tsv', Cruncher())
    visualization = Visualizer(preprocessor).visualize(labels = ['positive'],method = 'frame')

    model = Vectorizer(preprocessor).vectorize(labels = ['positive'])
    
    Visualizer(preprocessor).tsne(model)