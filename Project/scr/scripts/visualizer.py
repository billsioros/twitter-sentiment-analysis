
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud, STOPWORDS

class Visualizer:

    def __init__(self, preprocessor):

        self.preprocessor = preprocessor


    def visualize(self, labels=['positive', 'negative', 'neutral'], method='cloud'):
        
        tokens = []

        for label in set(labels):
            if label not in self.preprocessor.tweets.keys():
                raise ValueError("'" + label + "' is not a valid label")
            else:
                tokens += [token for tweet in self.preprocessor.tweets[label] for token in tweet]
        
        if method == 'cloud':
            return self.cloud(tokens)
        elif method == 'frame':
            return self.frame(tokens)
        else:
            raise ValueError("'" + method + "' is not supported")


    @staticmethod
    def frame(tokens):
        
        count = Counter(tokens)

        dataFrame = pd.DataFrame(data=count.most_common(50), columns=['Word', 'Count'])

        print(dataFrame)

        return dataFrame


    @staticmethod
    def cloud(tokens):

        wordcloud = WordCloud(width = 1200, height = 1200,
                background_color ='white',
                stopwords = set(STOPWORDS),
                min_font_size = 14).generate(' '.join(tokens))

        plt.figure(figsize = (8, 8), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)
        plt.show()

        return wordcloud

