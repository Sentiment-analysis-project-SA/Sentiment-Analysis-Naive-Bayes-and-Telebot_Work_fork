import numpy as np
import time
import sklearn
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

import pickle
from words_proc import *
from time import time
from my_models import *
from data_load import *
    
def load():
    texts = []
    y = []
    loader = DataLoader()
    for sample in loader:
        texts.append(tokenize(sample.x))
        y.append(int(sample.y))
    return texts, y
  
  
def time_count(func):
    def wrapper(*args, **kwargs):
        t1 = time()
        func(*args,  **kwargs)
        t2 = time()
        print("Executed in %i" % (t2-t1))
    return wrapper
  
  
@time_count
def model_training():
    model.fit(X_train, y_train)
    print("Accuracy is %f" % accuracy_score(model.predict(X_test), y_test))
    #SAVING RESULTS
    with open('model.data', 'wb') as f:
        pickle.dump(model, f)
    with open('words.data', 'wb') as f:
        pickle.dump(words, f)

        
texts, y = load()

label_encoding = False
if label_encoding:
    X, words = my_label_encoding(texts, y)
else:
    X, words = my_vectorizer(texts, y)

models = {0: GaussianNB(), 1: NaiveBayes()}
num = 1
model = models[num]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                    random_state  = 76)
model_training()
