from sklearn.cluster import SpectralClustering
import numpy as np


def Spectral_Cluster(input_data, clusters):

    clustering = SpectralClustering(n_clusters=clusters, assign_labels='discretize', random_state=0).fit(input_data)

    

           # A Dictionary to Store the Cluster IDs & the Respective Data with IDs
    kk = {}
    temp = []

    # Creating the Key for the number of Clusters as mentioned
    for k in range(len(np.unique(clustering.labels_))):
        kk[k] = []

    # Stores the Cluster ID's for Reference & Assignment 
    tt = list(kk.keys())

    # Assigning Data elements to their respective Clusters 
    for t in range(len(tt)):
        for i in range(len(clustering.labels_)):
            if clustering.labels_[i] == tt[t]:
                kk[t]+=[i]
    
    return kk, clustering.labels_