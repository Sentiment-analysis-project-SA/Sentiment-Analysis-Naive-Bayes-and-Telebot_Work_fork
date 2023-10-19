import numpy as np
import collections


class NaiveBayes:
    """ Just simple Naive Bayes classifier, which uses simple Bayes' theorem to classify and nothing more beyond it.
        Methods:
            fit(X :np.array, y: list) - to train on data.
            predict(X: np.array) - to predict to which classes corresponds X.
    """
    def __init__(self):
        self.apriori = None
        self.freq = None

    def fit(self, X: np.array, y: list):
        if self.apriori is None and self.freq is None:
            self.apriori = np.array(list(dict(sorted(collections.Counter(y).items())).values()))
            self.freq = np.zeros(shape = (len(self.apriori), len(X[0])))

        for sample, class_num in zip(X, y):
            self.freq[class_num] += sample
        #TODO: add recount appriori for new data or maybe not

        self.freq = np.transpose(np.transpose(self.freq) / self.apriori)
        self.apriori = self.apriori / self.apriori.sum()
        
    def predict(self, X: np.array):
        result = np.zeros(shape = (len(X)))
        for i in range(len(X)):
            freq = self.freq[:, X[i] != 0.0]
            freq[freq == 0.0] = 0.000000000001
            result[i] = ((freq * X[i][X[i] != 0.0]).prod(axis = 1) * self.apriori).argmax()
        return result

    def __repr__(self):
        return "NaiveBayes()"
