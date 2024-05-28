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
from sklearn.preprocessing import StandardScaler

clfs = {
    'GNB': GaussianNB(),
    'DT': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(),
}
metric = balanced_accuracy_score
alpha = 0.05

X, y = load_digits(return_X_y=True)

ss = StandardScaler()

kf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=2137)
results = np.zeros((10, len(clfs)))
for fold, (train, test) in tqdm(enumerate(kf.split(X,y))):
    for clf_idx, clfn in enumerate(clfs):
        clf = clone(clfs[clfn])
        ss.fit(X[train])
        X_train = ss.transform(X[train])
        X_test = ss.transform(X[test])
        clf.fit(X_train, y[train])
        y_pred = clf.predict(X_test)
        score = metric(y[test], y_pred)
        results[fold, clf_idx] = score

mean_results = np.round(np.mean(results, axis=0), decimals=3)
std_results = np.round(np.std(results, axis=0), decimals=3)
print(clfs)
print(mean_results)
print(std_results)