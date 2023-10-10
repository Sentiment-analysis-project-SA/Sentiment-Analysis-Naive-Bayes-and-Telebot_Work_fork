import numpy as np

def majority(y_pred):
    pass

class Ensemble:
    def __init__(self, models, classes = 3):
        self.models = models
        self.classes = classes

    def fit(self, X: np.array, y: list):
        for model in self.models:
            print(model)
            model.fit(X, y)

    def predict(self, X: np.array):
        """Simple majority for now"""
        result = np.zeros(shape = (len(self.models), len(X)))
        for i, model in enumerate(self.models):
            result[i] = model.predict(X)
        result2 = np.zeros(shape=(len(X), self.classes))
        for i in range(len(X)):
            for j in range(len(self.models)):
                result2[i][int(result[j][i])] += 1

        return result2.argmax(axis = 1)


