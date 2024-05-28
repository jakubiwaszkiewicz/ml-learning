# zad 1
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=700, n_features=2,n_redundant=0, n_clusters_per_class=1, flip_y=0.08)

data = np.concatenate((X, y.reshape(-1,1)), axis=1)


plt.scatter(X[:,0],X[:,1], c=y)
plt.savefig("dataset.png")
plt.grid(True)
np.savetxt("zad1.csv", data)