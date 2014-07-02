import sys, os
import numpy as np
import string
from operator import itemgetter as ig
from sklearn.linear_model import LogisticRegression as LR

vocab = []  # the features used in the classifier


def buildvocab(N):
    vocab = []
    stopwords = open('stopwords.txt').read().lower().split()

    vocab1 = {}

    # TODO: Populate vocab list with N most frequent words in training data, minus stopwords
    root = os.getcwd()
    root += '/pos'
    for file in os.listdir(root):
        if file.endswith(".txt"):
            with open(root + '/' + file, 'r') as f:
                for line in f:
                    words = line.lower().split(' ')
                    for word in words:
                        if word not in stopwords and word not in string.punctuation:
                            if word not in vocab1:
                                vocab1[word] = 0
                            else:
                                vocab1[word] += 1


    # need to cut vocab list down to N most frequent words
    vocab = [0 for i in range(N+1)]
    limiter = 0
    while len(vocab) > N:
        vocab = [(k, v) for k, v in vocab1.items() if v > limiter]
        print len(vocab)
        limiter += 1
    return vocab


def vectorize(fn):
    vocab = fn
    vector = np.zeros(len(vocab))

    # TODO: Create vector representation of
    vector = np.array([i[1] for i in vocab])

    return vector, vocab


def make_classifier():

    # TODO: Build X matrix of vector representations of review files, and y vector of labels

    lr = LR()
    lr.fit(X, y)

    return lr


def test_classifier(lr):
    global vocab
    test = np.zeros((len(os.listdir('test')),len(vocab)))
    testfn = []
    i = 0
    y = []
    for fn in os.listdir('test'):
        testfn.append(fn)
        test[i] = vectorize(os.path.join('test',fn))
        ind = int(fn.split('_')[0][-1])
        y.append(1 if ind == 3 else -1)
        i += 1

    assert(sum(y)==0)
    p = lr.predict(test)

    r, w = 0, 0
    for i, x in enumerate(p):
        if x == y[i]:
            r += 1
        else:
            w += 1
            print(testfn[i])
    print(r, w)


if __name__ == '__main__':
    buildvocab()
    lr = make_classifier()
    test_classifier(lr)
