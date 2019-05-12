
import re

from sklearn.metrics import f1_score, accuracy_score

import numpy as np

class Evaluator:

    def __init__(self, filename='..\\..\\twitter_data\\SemEval2017_task4_subtaskA_test_english_gold.txt'):

        with open(filename, mode='r', encoding='ascii', errors='ignore') as file:
            self.results = {}

            for line in file.readlines():
                id, result = line.split()

                self.results[id] = result

    def evaluate(self, unknown, method='f1score'):

        method = re.sub(r'''_|-|\ ''', '', method)

        if not isinstance(unknown, dict):
            raise ValueError("'unknown' is not an instance of 'dict'")

        facts = [self.results[id] for id in unknown.keys()]

        preds = list(unknown.values())

        if method == 'f1score':
            return f1_score(facts, preds, average='weighted', labels=np.unique(preds))
        elif method == 'accuracyscore':
            return accuracy_score(facts, preds, normalize=True)
        else:
            raise ValueError("'" + method + "' is not supported")

