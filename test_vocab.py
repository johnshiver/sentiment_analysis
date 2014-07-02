import os
import string

# build vocabulary
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
