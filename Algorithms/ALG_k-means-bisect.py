# This function performs Clustering on the Data; 

from sklearn.cluster import BisectingKMeans

def K_MEANS_BISECT(input_data, cluster):   # The Function requires input Data in the format of (Data, Number of Clusters) 
    # Note: the Data should be read from Datareader or should be converted to the format of (Number of elements, Dimension in 1D) as an array. e.g. If there are 20 files of data with dimensions of 256*256, the Input should be given in theform of 20 files with 1Dimension of 65,536

    # K_Means algorithm
    Kmean_BISECT = BisectingKMeans(n_clusters = cluster, random_state=0)#, n_init = 100)
    
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

    