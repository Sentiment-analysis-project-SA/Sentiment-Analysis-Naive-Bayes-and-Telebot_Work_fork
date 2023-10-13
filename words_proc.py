from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import copy

def tokenize(message_text):
    line = message_text.lower()
    res = line.replace('.', '').replace('?','').replace('!', '').split()
    stem = WordNetLemmatizer()
    res1 = res
    res = []
    for word in res1:
        res.append(stem.lemmatize(word))
    return res

def stop_word(word):
    return word in stopwords.words("english")


def my_vectorizer(texts, y):
    words, count = {}, {}
    total_count, class_count = {}, {}
    temp = 0
    class_num = 3  # TODO: ADD COUNTING OF UNIC CLASSES INSIDE y
    for i in range(len(texts)):
        for word in texts[i]:
            if (word not in class_count):
                class_count[word] = [0 for c in range(class_num)]
                class_count[word][y[i]] = 1  # MARK THAT WORD APPEARS IN THIS CLASS
            else:
                class_count[word][y[i]] += 1

    count = {k: sum(class_count[k]) for k in class_count.keys()}
    total_count = copy.deepcopy(count)
    for word in class_count.keys():
        class_count[word] = [class_count[word][i] / total_count[word] for i in range(len(class_count[word]))]

    count = {word: c for word, c in count.items() if c >= 20
             if stop_word(word) == False
             if max(class_count[word]) >= 0.45
             if min(class_count[word]) <= 0.25
             }  # deletion words under these criterias

    words = {word: i for i, word in enumerate(count.keys())}
    result = np.zeros(shape=(len(texts), len(words)), dtype='i4')
    threshold = 0.45
    for i in range(len(texts)):
        for word in texts[i]:
            if (word in words):
                class_num = y[i]
                if (class_count[word][class_num] < threshold):
                    result[i][words[word]] = 0
                else:
                    result[i][words[word]] += 1
    return result, words


def my_label_encoding(texts, y):
    unique = {}
    ec = {}  # 0 - padding
    temp = 1
    for line in texts:
        for word in line:
            if word not in unique:
                unique[word] = 1
                ec[word] = temp
                temp = temp + 1
            else:
                unique[word] += 1

    maxlen = max([len(texts[i]) for i in range(len(texts))])
    X = np.zeros(shape=(len(texts), maxlen))
    for i in range(len(texts)):
        for j in range(len(texts[i])):
            X[i][j] = ec[texts[i][j]]

    return X, ec