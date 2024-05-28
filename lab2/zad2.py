#zad 2
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=700, n_features=2, n_redundant=0, n_clusters_per_class=1, flip_y=0.08, random_state=1410)


GNB = GaussianNB()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1410)



metric = accuracy_score

GNB.fit(X_train, y_train)
y_pred=GNB.predict_proba(X_test)

y_pred = np.argmax(y_pred, axis=1)

score = metric(y_test, y_pred)

fig, ax = plt.subplots(1, 2, figsize=(7, 7))

plt.suptitle(f"Dokładność estymacji: {np.round(score, 3)}")

ax[0].scatter(X_test[:,0],X_test[:,1],c=y_test)
ax[1].scatter(X_test[:,0],X_test[:,1],c=y_pred)

plt.savefig("zad2")