
import os
import sys
import re
import string

from nltk import word_tokenize
from nltk.corpus import stopwords

from util import platform

class Preprocessor:

    valid_labels = { 'positive', 'negative', 'neutral', 'unknown' }

    urlregex = r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))'''
    tagregex = r'''@[^\s]+'''

    def __init__(self, filename, cruncher, save=True):

        filename = platform.path(filename)

        if not os.path.isdir('out'):
            os.mkdir('out')

        self.filename, _ = os.path.splitext(os.path.basename(filename))

        self.filename = os.path.join(os.path.curdir, 'out', self.filename)

        self.labels = {}

        self.tweets = {}

        for label in self.valid_labels:

            if not os.path.isfile(self.filename + '_' + label + '.tsv'):

                with open(filename, mode='r', encoding='utf8') as file:
                    lines = file.readlines()

                ignore = set(stopwords.words('english'))

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
                        self.tweets[tokens[0]] = tokens[3:]
                        self.labels[tokens[0]] = tokens[2]
                    else:
                        raise ValueError("'" + tokens[2] + "' is not a valid label")

                if save:
                    for valid_label in self.valid_labels:
                        filename = self.filename + '_' + valid_label + '.tsv'

                        tweets = { id: self.tweets[id] for id, label in self.labels.items() if label == valid_label }

                        print('<LOG>: Saving', str(len(tweets)).rjust(5), ("'" + valid_label + "'").ljust(12), 'tweets to', filename, file=sys.stderr)

                        with open(filename, 'w', encoding='ascii') as file:
                            file.write('\n'.join([id + '\t' + valid_label + '\t' + ' '.join(tweet) for id, tweet in tweets.items()]))

                return

        for valid_label in self.valid_labels:
            with open(self.filename + '_' + valid_label + '.tsv', mode='r', encoding='ascii') as file:
                tokenized_lines = [word_tokenize(line) for line in file.readlines()]

                for line in tokenized_lines:
                    self.tweets[line[0]] = line[2:]
                    self.labels[line[0]] = line[1]

                print('<LOG>: Loaded', str(len(tokenized_lines)).rjust(5), ("'" + valid_label + "'").ljust(12), 'tweets', file=sys.stderr)


    def by_label(self, labels):

        labels = set(labels)

        for label in labels:
            if label not in self.valid_labels:
                raise ValueError("'" + label + "' is not a valid label")

        return { label: [(id, self.tweets[id]) for id, label in self.labels.items() if label in labels ]}

