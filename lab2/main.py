import numpy as np

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score

testData=[[1,1,1,1], 1, [1,1,2,1], 1, [1,2,2,2], 2]

clfs = {
    'GNB': GaussianNB()
}