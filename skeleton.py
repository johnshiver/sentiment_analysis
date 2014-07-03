import sys, os
import numpy as np
import string
from collections import OrderedDict
from operator import itemgetter as ig
from sklearn.linear_model import LogisticRegression as LR

vocab = []  # the features used in the classifier


def _build_stopwords():
    stopwords = set(open('stopwords.txt').read().lower().split())
    bad_chars = set(['\n', "--"])
    stopwords = stopwords.union(bad_chars)
    stopwords = stopwords.union(set([i for i in string.punctuation]))
    return stopwords


def buildvocab(N):
    vocab = []
    vocab1 = {}
    # TODO: Populate vocab list with N most frequent words in training data, minus stopwords
    roots = ['/pos', '/neg']
    stopwords = _build_stopwords()

    for i in roots:
        root = os.getcwd() + i
        for file in os.listdir(root):
            if file.endswith(".txt"):
                with open(root + '/' + file, 'r') as f:
                    for line in f:
                        words = line.lower().split(' ')
                        for word in words:
                            if word not in stopwords:
                                if word not in vocab1:
                                    vocab1[word] = 0
                                else:
                                    vocab1[word] += 1

    # need to cut vocab list down to N most frequent words
    vocab = [0 for i in range(N+1)]
    limiter = 0
    while len(vocab) > N:
        vocab = [k for k, v in vocab1.items() if v > limiter]
        print len(vocab)
        limiter += 1
    ordered_vocab = OrderedDict()
    for i, word in enumerate(vocab):
        ordered_vocab[word] = i
    return ordered_vocab


def vectorize(review, vocab):
    vector = np.zeros(len(vocab))

    # TODO: Create vector representation of
    for word in review:
        if word in vocab:
            vector[vocab[word]] += 1
    return vector


def make_classifier(vocab):
    cwd = os.getcwd()
    pos = np.ones(len(os.listdir(cwd + '/pos')))
    neg = -1 * np.ones(len(os.listdir(cwd + '/neg')))
    y = np.concatenate((pos, neg))
    # TODO: Build X matrix of vector representations of review files, and y vector of labels
    # X = np.zeros((y.shape()[0], len(vocab)))
    X = np.array([])
    stopwords = _build_stopwords()
    roots = ['/pos', '/neg']

    review = []
    for i in roots:
        root = cwd + i
        for file_ in os.listdir(root):
            print file_
            if file_.endswith(".txt"):
                with open(root + '/' + file_, 'r') as f:
                    for line in f:
                        words = line.lower().split(' ')
                        for word in words:
                            if word not in stopwords and word in vocab:
                                review.append(word)
                X = np.concatenate((X, vectorize(review, vocab)))
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
    vocab = buildvocab(100)
    lr = make_classifier(vocab)
    test_classifier(lr)
