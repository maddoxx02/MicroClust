from sklearn.cluster import BisectingKMeans

def K_MEANS_BISECT(input_data, cluster):   # The Function requires input Data in the format of (Data, Number of Clusters) 
    

    # K_Means algorithm
    Kmean_BISECT = BisectingKMeans(n_clusters = cluster,init = 'k-means++', random_state=0)#, n_init = 100)
    
    # Fitting It to the Data
    Kmean_BISECT.fit(input_data)
    
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
        for i in range(len(Kmean_BISECT.labels_)):
            if Kmean_BISECT.labels_[i] == tt[t]:
                kk[t]+=[i]

    # Returning a Dictionary with The Number of Clusters and respective Elements within
    return kk, Kmean_BISECT.labels_