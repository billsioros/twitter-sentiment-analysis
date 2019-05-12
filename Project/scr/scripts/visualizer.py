
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from wordcloud import WordCloud, STOPWORDS
import numpy as np

from gensim.models import Word2Vec

from cruncher import Cruncher
from preprocessor import Preprocessor
from dictionary import Dictioanry
from vectorizer import Vectorizer

np.random.seed(19680801)

class Visualizer:

    supported_methods = { 'word_cloud', 'bar_plot', 'tsne', 'heat_map' }

    def __init__(self, preprocessor):

        if not isinstance(preprocessor, Preprocessor):
            raise ValueError("'preprocessor' is not an instance of 'Preprocessor'")

        self.preprocessor = preprocessor


    def visualize(self, labels=Preprocessor.valid_labels, method='word_cloud', dictionary=None, model=None, max_words=300):

        tokens = []

        for _, tweets in self.preprocessor.by_label(labels).items():
            for _, tweet in tweets:
                tokens += [token for token in tweet]

        if method == 'word_cloud':
            self.word_cloud(tokens)
        elif method == 'bar_plot':
            self.bar_plot(tokens)
        elif method == 'tsne':
            self.tsne(model, max_words)
        elif method == 'heat_map':
            self.heat_map(tokens, dictionary)
        else:
            raise ValueError("'" + method + "' is not supported")


    @staticmethod
    def bar_plot(tokens):

        count = Counter(tokens)

        dataFrame = pd.DataFrame(data=count.most_common(50), columns=['Word', 'Count'])

        dataFrame.plot.bar(x='Word',y='Count',figsize = (20,10))


    @staticmethod
    def word_cloud(tokens):

        wordcloud = WordCloud(width = 1200, height = 1200,
                background_color ='white',
                stopwords = set(STOPWORDS),
                min_font_size = 14).generate(' '.join(tokens))

        plt.figure(figsize = (8, 8), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        plt.show()


    @staticmethod
    def tsne(model, max_words):

        if not isinstance(model, Word2Vec):
            raise ValueError("'model' is not an instance of 'Word2Vec'")

        if not isinstance(max_words, int) or max_words <= 0:
            raise ValueError("'max_words' must have an integer value greater than 0")

        labels = []
        tokens = []
        counter = 0

        for word in model.wv.vocab:
            tokens.append(model.wv[word])
            labels.append(word)
            counter +=1
            if counter == max_words:
                break

        tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=5000, random_state=23,)
        new_values = tsne_model.fit_transform(tokens)

        x = []
        y = []

        for value in new_values:
            x.append(value[0])
            y.append(value[1])

        plt.figure(figsize=(16, 16))
        for i in range(len(x)):
            plt.scatter(x[i],y[i])
            plt.annotate(labels[i],
                xy=(x[i], y[i]),
                xytext=(5, 2),
                textcoords='offset points',
                ha='right',
                va='bottom')

        plt.show()


    @staticmethod
    def heat_map(tokens, dictioanry):

        if not isinstance(dictioanry, Dictioanry):
            raise ValueError("'dictioanry' is not an instance of 'Dictioanry")

        rgb = lambda valence: (255 + Dictioanry.convert(valence, (-1, 1), (0, 255)) + 255) / 3

        x, y, c = [], [], []

        for token in tokens:
            if token in dictioanry.valences:
                x.append(np.random.rand())
                y.append(np.random.rand())
                c.append(rgb(np.mean(dictioanry.valences[token])))

        plt.scatter(x, y, c=c, alpha=0.8)

        plt.show()


if __name__ == "__main__":

    preprocessor = Preprocessor(['train.tsv', 'test.tsv'], Cruncher())

    dictionary = Dictioanry('..\\..\\lexica')

    vectorizer = Vectorizer()

    labels, vectors = vectorizer.vectorize(preprocessor, dictionary)

    visualizer = Visualizer(preprocessor)

    for method in Visualizer.supported_methods:

        visualizer.visualize(method=method, dictionary=dictionary, model=vectorizer.underlying)

