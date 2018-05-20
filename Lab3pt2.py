"""
Created on Tue Feb 27 18:59:22 2018

@author: victo
"""
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN as DBS
from sklearn.cluster import AgglomerativeClustering as AC
import matplotlib.pyplot as plt

def entropy(X):
    f = np.bincount(X)
    f = f[f>0]
    p = f/np.sum(f)
    ent = -np.sum(p*np.log2(p))
    return ent

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
  
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1 
    
X_people = people.data[mask]
y_people = people.target[mask]
X_people = X_people / 255.

k = 100
km = KMeans(n_clusters = k)
km.fit(X_people)
km_assignments = km.labels_
km_means = km.cluster_centers_

for i in range(k):
    ind = km_assignments == i
    ent = entropy(y_people[ind])
    print("Cluster {:d}, size = {:d}, entropy = {:.3f}".format(i, np.sum(ind), ent))
    if ent > 4 and np.sum(ind) > 10:        
        fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
        for i, (component, ax) in enumerate(zip(y_people[ind], axes.ravel())):
            ax.imshow(component.reshape(image_shape), cmap = 'grey')
            ax.set_title("{}. component".format(i+1))
            
dbs = DBS()
dbs.fit(X_people)
dbs_assignments = dbs.labels_
dbs_means = dbs.core_sample_indices_

print(len(dbs_means))

a = 100
ac = AC(n_clusters = a)
ac.fit(X_people)
ac_assignments = ac.labels_
ac_means = ac.n_clusters
print(ac_means)

for i in range(a):
    ind = ac_assignments == i
    ent = entropy(y_people[ind])
    print("Cluster {:d}, size = {:d}, entropy = {:.3f}".format(i, np.sum(ind), ent))
    if ent > 4 and np.sum(ind) > 10:        
        fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
        for i, (component, ax) in enumerate(zip(y_people[ind], axes.ravel())):
            ax.imshow(component.reshape(image_shape), cmap = 'grey')
            ax.set_title("{}. component".format(i+1))