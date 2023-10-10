import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def majority(y_pred):
    pass

class Ensemble:
    def __init__(self, models, classes = 3):
        self.models = models
        self.classes = classes
        self.rules = ["majority", "weighted_majority"]
        #TODO: add bootstrap aggregation as rule
    def fit(self, X: np.array, y: list, rule:str = "majority"):
        """Rules are: majority, weighted_majority"""
        X_test, X_val, y_test, y_val = train_test_split(X, y, test_size=1/8,
                                                        random_state=76)
        match rule:
            case "majority":
                for model in self.models:
                    print(model)
                    model.fit(X, y)
            case "weighted_majority":
                for model in self.models:
                    print(model)
                    model.fit(X, y)
                self.weights = np.zeros(shape = (len(self.models)))
                for i, model in enumerate(self.models):
                    self.weights[i] = accuracy_score(model.predict(X_val), y_val)
            case _:
                raise Exception("No such rule present")

        self.X_val = X_val
        self.y_val = y_val
        self.rule = rule


    def predict(self, X: np.array):
        """Simple majority for now"""
        match self.rule:
            case "majority":
                result = np.zeros(shape=(len(self.models), len(X)))
                for i, model in enumerate(self.models):
                    result[i] = model.predict(X)
                result2 = np.zeros(shape=(len(X), self.classes))
                for i in range(len(X)):
                    for j in range(len(self.models)):
                        result2[i][int(result[j][i])] += 1
                return result2.argmax(axis=1)

            case "weighted_majority":
                result = np.zeros(shape=(len(self.models), len(X)))
                for i, model in enumerate(self.models):
                    result[i] = model.predict(X)
                result2 = np.zeros(shape=(len(X), self.classes))
                for i in range(len(X)):
                    for j in range(len(self.models)):
                        result2[i][int(result[j][i])] += self.weights[j]
                return result2.argmax(axis=1)



