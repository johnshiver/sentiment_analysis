from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import fetch_20newsgroups_vectorized


newsgroups = fetch_20newsgroups()
newsgroups.keys()

newsgroups.target_names

# remove metadata to prevent over-fitting of these features
data = fetch_20newsgroups_vectorized(remove=('headers', 'footers', 'quotes'))

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=0.1)
clf.fit(data['data'], data['target'])

predicted = clf.predict(data.data)
print(clf.score(data['data'], data['target']))

from sklearn.cross_validation import cross_val_score
cross_val_score(clf, data.data, data.target, cv=10)

alphas = [1E-4, 1E-3, 1E-2, 1E-1]
results = []
for alpha in alphas:
    clf = MultinomialNB(alpha)
    results.append(np.mean(cross_val_score(clf, data.data, data.target)))
results