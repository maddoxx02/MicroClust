#K means ++

from sklearn.cluster import KMeans

# have to make A bigger selection menu for types of data Input: 
# 1. 

def K_MEANS_PLUS(input_data, cluster):   

    # K_Means algorithm
    Kmean = KMeans(n_clusters = cluster, init = 'k-means++', random_state=0)#, n_init = 100)
    
    # Fitting It to the Data
    Kmean.fit(input_data)
    
    # A Dictionary to Store the Cluster IDs & the Respective Data with IDs
    kk = {}
    temp = []

    # Creating the Key for the number of Clusters as mentioned
    for k in range(cluster):
        kk[k] = []

    # Stores the Cluster ID's for Reference & Assignment 
    tt = list(kk.keys())

    # Assigning Data elements to their respective Clusters 
    for t in range(len(tt)):
        for i in range(len(Kmean.labels_)):
            if Kmean.labels_[i] == tt[t]:
                kk[t]+=[i]

    # Returning a Dictionary with The Number of Clusters and respective Elements within
    return kk, Kmean.labels_