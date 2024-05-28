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
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA


clfs = {
    'GNB': GaussianNB(),
    'DT': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(),
}

clfs_names = {k:None for k in clfs}

metric = balanced_accuracy_score
alpha = 0.05

X, y = load_digits(return_X_y=True)

ss = StandardScaler()
kbest = SelectKBest(k=8)
pca = PCA(n_components=0.8)
kf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=2137)
results = np.zeros((10, len(clfs)))
print("")
print("")
print("")
print("KBest")
for fold, (train, test) in enumerate(kf.split(X,y)):
    for clf_idx, clfn in enumerate(clfs):
        clf = clone(clfs[clfn])
        X_train = kbest.fit_transform(X[train], y[train])
        X_test = kbest.transform(X[test])
        clf.fit(X_train, y[train])
        y_pred = clf.predict(X_test)
        score = metric(y[test], y_pred)
        results[fold, clf_idx] = score

mean_results = np.round(np.mean(results, axis=0), decimals=3)
std_results = np.round(np.std(results, axis=0), decimals=3)

print(list(clfs.keys()))
print(mean_results)
print(std_results)

print("")
print("PCA")
for fold, (train, test) in enumerate(kf.split(X,y)):
    for clf_idx, clfn in enumerate(clfs):
        clf = clone(clfs[clfn])
        pca.fit(X[train])
        X_train = pca.transform(X[train])
        X_test = pca.transform(X[test])
        clf.fit(X_train, y[train])
        y_pred = clf.predict(X_test)
        score = metric(y[test], y_pred)
        results[fold, clf_idx] = score

mean_results = np.round(np.mean(results, axis=0), decimals=3)
std_results = np.round(np.std(results, axis=0), decimals=3)

print(list(clfs.keys()))
print(mean_results)
print(std_results)

print("")
print("Norm")

for fold, (train, test) in enumerate(kf.split(X,y)):
    for clf_idx, clfn in enumerate(clfs):
        clf = clone(clfs[clfn])
        ss.fit(X[train])
        X_train = ss.transform(X[train])
        X_test = ss.transform(X[test])
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        score = metric(y[test], y_pred)
        results[fold, clf_idx] = score

mean_results = np.round(np.mean(results, axis=0), decimals=3)
std_results = np.round(np.std(results, axis=0), decimals=3)

print(list(clfs.keys()))
print(mean_results)
print(std_results)

print("")
print("Base")

for fold, (train, test) in enumerate(kf.split(X,y)):
    for clf_idx, clfn in enumerate(clfs):
        clf = clone(clfs[clfn])
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        score = metric(y[test], y_pred)
        results[fold, clf_idx] = score

mean_results = np.round(np.mean(results, axis=0), decimals=3)
std_results = np.round(np.std(results, axis=0), decimals=3)

print(list(clfs.keys()))
print(mean_results)
print(std_results)