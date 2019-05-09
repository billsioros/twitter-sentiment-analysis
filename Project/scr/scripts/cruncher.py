
from nltk.stem import  WordNetLemmatizer
from nltk.stem import PorterStemmer

class Cruncher:

    def __init__(self, method='lemmatizer'):

        if method == 'lemmatizer':
            self.underlying = WordNetLemmatizer()
            self.crunch = self.underlying.lemmatize
        elif method == 'stemmer':
            self.underlying = PorterStemmer()
            self.crunch = self.underlying.stem
        else:
            raise ValueError("'" + method + "' is not supported")

