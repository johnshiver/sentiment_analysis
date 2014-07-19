#! /usr/bin/python
import numpy
from sklearn.datasets import fetch_20newsgroups_vectorized
from IPython import parallel
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score


def run_MultnomialNB(alpha):
    u"""Return cross validation score of Multinomial NB classifier."""
    results = []
    clf = MultinomialNB(alpha)
    results.append(numpy.mean(cross_val_score(clf, DATA.data, DATA.target)))
    return results


def parallelize_MultnomiaNB(alphas):
    u"""Return the highest accuracy and corresponding alpha."""
    # Create handler for parallel processes
    print u"Creating clients."
    clients = parallel.Client()

    # Create views to interact with clients
    dview = clients.direct_view()
    lview = clients.load_balanced_view()
    lview.block = True

    # Apply imports to clients
    with dview.sync_imports():
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.cross_validation import cross_val_score
        import numpy

    # Fetch data and remove metadata to prevent over-fitting of these features
    print u"Fetching data."
    DATA = fetch_20newsgroups_vectorized(
        remove=('headers', 'footers', 'quotes')
        )

    # Distribute alpha and data
    print u"Distributing data."
    dview.scatter('alpha', alphas)
    dview['DATA'] = DATA

    # Compile results
    print u"Compiling results."
    res = lview.map(run_MultnomialNB, alphas)

    # Flatten the list and find the maximum accuracy and associated alpha
    max_acc, best_alpha = max(zip(sum(res, []), alphas))
    return round(max_acc, 4) * 100, best_alpha

alphas = [round(0.0001 + (x * 0.0001), 4) for x in xrange(200)]
accuracy, alpha = parallelize_MultnomiaNB(alphas)

ret_str = u"The highest accuracy is {}% with an alpha of {}"
print ret_str.format(accuracy, alpha)
