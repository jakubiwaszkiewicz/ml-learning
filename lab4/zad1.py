
# importing os module  
from os import listdir
import numpy as np
from numpy import loadtxt
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from scipy.stats import ttest_rel
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Get the list of all files and directories 
path = "datasets"
dir_list = listdir(path) 
clfs = {
    'GNB': GaussianNB(),
    'MLP': MLPClassifier(),
    'KNN': KNeighborsClassifier(),
}

kf = RepeatedStratifiedKFold(n_repeats=5, n_splits=2, random_state=713)
results = np.zeros((len(dir_list),5*2,len(clfs)))


for id, csv in tqdm(enumerate(dir_list)):
    dataset = loadtxt(f"datasets/{csv}", delimiter=",")
    y = dataset[:, -1] # for last column
    X = dataset[:, :-1] # for all but last column
    
    metric = accuracy_score
    stest=ttest_rel
    alpha=.5

    for fold, (train, test) in enumerate(kf.split(X, y)):
        for clf_idx, clfn in enumerate(clfs):
            clf = clone(clfs[clfn])
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            score=metric(y[test], y_pred)
            results[id, fold, clf_idx] = score
            
np.save('results', results)