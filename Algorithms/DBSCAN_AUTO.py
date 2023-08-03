# https://www.kdnuggets.com/2022/08/implementing-dbscan-python.html 
# https://towardsdatascience.com/machine-learning-clustering-dbscan-determine-the-optimal-value-for-epsilon-eps-python-example-3100091cfbc
# https://stats.stackexchange.com/questions/88872/a-routine-to-choose-eps-and-minpts-for-dbscan 

# Determining optimal EPS value: https://iopscience.iop.org/article/10.1088/1755-1315/31/1/012012/pdf

# Distance can be dependent on Elements or chemicals used for the polymer

# Minimal domain knowledge
#pip install kneed

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
from kneed import KneeLocator

def DBSCAN_AUTOEPS_Cluster(input_data):

    neigh = NearestNeighbors(n_neighbors=2)

    nbrs = neigh.fit(input_data)

    distances, indices = nbrs.kneighbors(input_data)

    distances = np.sort(distances, axis=0)
    distances = distances[:,1]

    x = range(0, len(distances))

    kn = KneeLocator(x, distances, curve='convex', direction='increasing')
    
    clustering = DBSCAN(eps = distances[kn.knee], min_samples = 1).fit(input_data)#random_state=0

    # A Dictionary to Store the Cluster IDs & the Respective Data with IDs
    kk = {}
    temp = []

    # Creating the Key for the number of Clusters as mentioned
    for k in range(max(clustering.labels_)):
        kk[k] = []

    # Stores the Cluster ID's for Reference & Assignment 
    tt = list(kk.keys())

    # Assigning Data elements to their respective Clusters 
    for t in range(len(tt)):
        for i in range(len(clustering.labels_)):
            if clustering.labels_[i] == tt[t]:
                kk[t]+=[i]

    # Returning a Dictionary with The Number of Clusters and respective Elements within
    return kk, clustering.labels_


