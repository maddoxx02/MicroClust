from sklearn.cluster import OPTICS

def OPTICS_A(input_data, min_samp): #max_eps = 475
    
    clustering = OPTICS(min_samples = min_samp, max_eps = 475)

    clustering.fit(input_data)

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

    # Returning a Dictionary with The Number of Clusters and respective Elements within
    return kk, clustering.labels_