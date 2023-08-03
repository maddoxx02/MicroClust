from sklearn.cluster import AffinityPropagation
def AF_Propagate(input_data):
    
    clustering_AFP = AffinityPropagation(random_state=0)#random_state=5
    
    clustering_AFP.fit(input_data)
    
        # A Dictionary to Store the Cluster IDs & the Respective Data with IDs
    kk = {}
    temp = []

    # Creating the Key for the number of Clusters as mentioned
    for k in range(max(clustering_AFP.labels_)):
        kk[k] = []

    # Stores the Cluster ID's for Reference & Assignment 
    tt = list(kk.keys())

    # Assigning Data elements to their respective Clusters 
    for t in range(len(tt)):
        for i in range(len(clustering_AFP.labels_)):
            if clustering_AFP.labels_[i] == tt[t]:
                kk[t]+=[i]

    # Returning a Dictionary with The Number of Clusters and respective Elements within
    return kk, clustering_AFP.labels_
