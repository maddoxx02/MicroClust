# https://pypi.org/project/fuzzy-c-means/ 
# requirments = pip install fuzzy-c-means 
from fcmeans import FCM


def FUZZY_C(input_data, cluster):


    fcm = FCM(n_clusters=cluster, random_state=0, max_iter = 300)

    fcm.fit(input_data)

    fcm_labels = fcm.predict(input_data)

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
        for i in range(len(fcm_labels)):
            if fcm_labels[i] == tt[t]:
                kk[t]+=[i]

    # Returning a Dictionary with The Number of Clusters and respective Elements within
    return kk, fcm_labels