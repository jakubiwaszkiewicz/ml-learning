#zad 2
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

X, y = make_classification(n_samples=700, n_features=2, n_redundant=0, n_clusters_per_class=1, flip_y=0.08, random_state=1410)



rsfk = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=1410)
scores = np.zeros((10))
for fold, (train, test) in enumerate(rsfk.split(X, y)):
    GNB = GaussianNB()
    GNB.fit(X[train], y[train])
    pred = GNB.predict(X[test])
    scores[fold] = accuracy_score(y[test], pred)

print("Nadzieja matematyczna", np.round(np.mean(scores), 3))
print("Odchylenie standardowe", np.round(np.std(scores), 3))
