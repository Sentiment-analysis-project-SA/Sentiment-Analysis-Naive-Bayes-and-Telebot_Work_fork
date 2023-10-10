import numpy as np
class Ensemble:
    def __init__(self, models):
        self.models = models

    def fit(self, X: np.array, y: list):
        for model in self.models:
            model.fit(X, y)

    def predict(self, X: np.array):
        pass
    