import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import ttest_rel
from sklearn.model_selection import RepeatedStratifiedKFold
from tqdm import tqdm
from sklearn.datasets import load_digits
from sklearn.base import clone
clfs = {
    'GNB': GaussianNB(),
    'DT': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(),
}
metric = balanced_accuracy_score
alpha = 0.05

X, y = load_digits(return_X_y=True)

kf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=2137)
results = np.zeros((10, len(clfs)))
for fold, (train, test) in tqdm(enumerate(kf.split(X,y)), total=len(clfs)):
    for clf_idx, clfn in enumerate(clfs):
        clf = clone(clfs[clfn])
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        score = metric(y[test], y_pred)
        results[fold, clf_idx] = score

mean_results = np.round(np.mean(results, axis=0), decimals=3)
std_results = np.round(np.std(results, axis=0), decimals=3)
print(clfs)
print(mean_results)
print(std_results)