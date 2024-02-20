from sklearn.cluster import HDBSCAN
from sklearn.datasets import load_digits


def HDBSCAN_MANUAL(input_data, min_c_s, min_samp, eps):

    clustering_HDB = HDBSCAN(min_cluster_size=min_c_s, min_samples = min_samp, cluster_selection_epsilon = eps, algorithm = 'brute', cluster_selection_method = 'eom')#5,3,475
    clustering_HDB.fit(input_data)


 # A Dictionary to Store the Cluster IDs & the Respective Data with IDs
    kk = {}
    temp = []

    # Creating the Key for the number of Clusters as mentioned
    for k in range(len(np.unique(clustering_HDB.labels_))):
        kk[k] = []

    # Stores the Cluster ID's for Reference & Assignment 
    tt = list(kk.keys())

    # Assigning Data elements to their respective Clusters 
    for t in range(len(tt)):
        for i in range(len(clustering_HDB.labels_)):
            if clustering_HDB.labels_[i] == tt[t]:
                kk[t]+=[i]

    # Returning a Dictionary with The Number of Clusters and respective Elements within
    return kk, clustering_HDB.labels_