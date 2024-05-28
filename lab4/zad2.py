import numpy as np
from scipy.stats import ttest_rel
data = np.load('results.npy')
print(data.shape)
stest = ttest_rel
t_stat = np.zeros((3, 3))
p_val = np.zeros((3, 3))
better = np.zeros((3, 3), dtype=bool)

data = data[8]
clfs = ['GNB', 'MLP', 'KNN']
for i in range(3):
    for j in range(3):
        clfn = clfs[i]
        clf_j = data[:,j]
        clf_i = data[:,i]
        t_stat_id, p_val_id = ttest_rel(clf_i, clf_j)
        t_stat[i, j] = t_stat_id
        p_val[i, j] = p_val_id
        better[i,j] = np.mean(clf_i) > np.mean(clf_j)
print(t_stat)
print(p_val)

print("t test studenta")
print(t_stat)
print("_______________")
print("p_val")
print(p_val)
print("_______________")
print("better")
print(better)
alpha = 0.05
stat_significant = p_val < alpha
print("_______________")
print("significatn matrix")
print(stat_significant)
print("_______________")
stat_better = stat_significant * better
print("stat better")
print(stat_better)
print("_______________")

for i in range(3):
    for j in range(3):
        if stat_better[i, j]:
            print(f'{clfs[i]} (dokładność={np.mean(data[:,i])}) jest lepszy statystycznie od {clfs[j]} (dokładność={np.mean(data[:,j])})')