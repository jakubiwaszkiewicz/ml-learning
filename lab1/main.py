import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

data = np.loadtxt("iris.csv",delimiter=",", dtype="object")

column_names=data[0]

data = data[1:]

labels=data[:,-1]

features=data[:,0:4]

labels[labels=='setosa'] = 0
labels[labels=='versicolor'] = 1
labels[labels=='virginica'] = 2

features = features.astype(float)

def zad1():
    fig, ax = plt.subplots(1,1,figsize=(7,7))
    ax.scatter(features[:,0],features[:,1],c=labels)
    plt.show()



def zad2():
    fig, ax = plt.subplots(4,4,figsize=(15,15))
    for i in range(4):
        for j in range(4):
            ax[i,j].scatter(features[:,j],features[:,i],c=labels)
            ax[i,j].set_xlabel(column_names[j])
            ax[i,j].set_ylabel(column_names[i])
    plt.tight_layout()
    plt.show()

def zad3():
    new_features = features[:,2:4]
    petal_lengths = new_features[:,:1]
    petal_widths = new_features[:,1:2]
    column_name = column_names[2:4]

    centroids=[]

    for i in range(0,len(np.unique(labels))):
        objects=new_features[labels==i]
        average = np.mean(objects,axis=0)
        centroids.append(average)


    centroids = np.array(centroids)
            
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.scatter(petal_lengths,petal_widths,c=labels, alpha=0.15)
    ax.scatter(centroids[:,0],centroids[:,1],c=[0,1,2], alpha=1)

    

    ax.scatter(3.1,1.2, c="black", marker="x")

    znaleziony_platek=[3.1,1.2]
    distances = cdist([znaleziony_platek], centroids, metric="euclidean")
    
    print("odleglosci:", distances)

    distances = np.argmin(distances)


    ax.set_xlabel(column_names[:1])
    ax.set_ylabel(column_names[1:2])
    plt.show()