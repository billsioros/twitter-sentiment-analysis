
import os
import sys
import re
import string

from nltk import word_tokenize
from nltk.corpus import stopwords

from util import platform
from cruncher import Cruncher

class Preprocessor:

    valid_labels = { 'positive', 'negative', 'neutral', 'unknown' }

    urlregex = r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))'''
    tagregex = r'''@[^\s]+'''

    def __init__(self, filenames, cruncher, save=True):

        if not isinstance(cruncher, Cruncher):
            raise ValueError("'" + cruncher + "' is not an instance of 'Cruncher'")

        if not os.path.isdir('out'):
            os.mkdir('out')

        self.path = platform.filename(filenames, ['preprocessed'])

        self.labels = {}

        self.tweets = {}

        if os.path.isfile(self.path + '.tsv'):

            with open(self.path + '.tsv', mode='r', encoding='ascii') as file:
                
                tokenized_lines = [word_tokenize(line) for line in file.readlines()]

                counts = dict(zip(self.valid_labels, [0] * len(self.valid_labels)))

                for line in tokenized_lines:

                    id, label, tokens = line[0], line[1], line[2:]

                    if tokens:
                        self.tweets[id] = tokens
                        self.labels[id] = label

                        counts[label] += 1

                for label, count in counts.items():
                    print('<LOG>: Loaded', str(count).rjust(5), ("'" + label + "'").ljust(max(map(len, self.valid_labels)) + 2), 'tweets from', self.path + '.tsv', file=sys.stderr)

                return
        

        for filename in filenames:

            with open(filename, mode='r', encoding='utf8') as file:
                print('<LOG>: Processing', ("'" + filename + "'").ljust(max(map(len, filenames)) + 2), file=sys.stderr)

                lines = file.readlines()

            ignore = set(stopwords.words('english'))

            counts = dict(zip(self.valid_labels, [0] * len(self.valid_labels)))

            for line in lines:
                # Remove any non ascii characters (for example emojis)
                line = line.encode('ascii', 'ignore').decode('ascii')

                # Remove any leading and trailing whitespace characters
                line = line.strip()

                # Convert every character to its lower case counterpart
                line = line.lower()

                # Remove any urls
                line = re.sub(self.urlregex, '', line)

                # Remove any tags
                line = re.sub(self.tagregex, '', line)

                # Remove any punctuation
                line = line.translate(str.maketrans('', '', string.punctuation))

                # Tokenize tweet at hand and lemmatize each one of its tokens
                # while removing any stopwords
                tokens = word_tokenize(line)

                tokens = [cruncher.crunch(token) for token in tokens if token not in ignore]

                if tokens[2] in self.valid_labels:

                    id, label, tokens = tokens[0], tokens[2], tokens[3:]

                    if tokens:
                        self.tweets[id] = tokens
                        self.labels[id] = label

                        counts[label] += 1

                else:
                    raise ValueError("'" + label + "' is not a valid label")

            for label, count in counts.items():

                print('<LOG>: Saving', str(count).rjust(5), ("'" + label + "'").ljust(max(map(len, self.valid_labels)) + 2), 'tweets to', self.path + '.tsv', file=sys.stderr)
                
            if save:

                with open(self.path + '.tsv', 'w', encoding='ascii') as file:
                    file.write('\n'.join([id + '\t' + self.labels[id] + '\t' + ' '.join(tweet) for id, tweet in self.tweets.items()]))


    def by_label(self, labels):

        labels = set(labels)

        for label in labels:
            if label not in self.valid_labels:
                raise ValueError("'" + label + "' is not a valid label")

        return { label: [(id, self.tweets[id]) for id, label in self.labels.items() if label in labels] }


if __name__ == "__main__":
    
    preprocessor = Preprocessor(['train.tsv', 'test.tsv'], Cruncher())

