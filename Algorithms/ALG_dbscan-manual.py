# NOTE

# SS = DBSCAN(eps=478, min_samples=3).fit(SF1) #EPS<=400 was noise & >=700 

# Woriking range: 460-480



from sklearn.cluster import DBSCAN
import numpy as np

def DBSCAN_MANUAL_Cluster(input_data, eps):
    
    clustering = DBSCAN(eps=eps, min_samples=3)# min samples = minimum samples to be considered as a Centroid
    
    clustering.fit(input_data)
    
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