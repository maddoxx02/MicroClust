import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import os


# status  is to produce Graphs & Save it. 
def HIERARCHY(input_data, clusters, STATUS):

    hierarchical_cluster = AgglomerativeClustering(n_clusters=clusters, affinity='euclidean', linkage='ward')
    labels = hierarchical_cluster.fit_predict(input_data)
    add = os.getcwd()
        # A Dictionary to Store the Cluster IDs & the Respective Data with IDs
    kk = {}
    temp = []

    # Creating the Key for the number of Clusters as mentioned
    for k in range(len(np.unique(labels))):
        kk[k] = []

    # Stores the Cluster ID's for Reference & Assignment 
    tt = list(kk.keys())

    # Assigning Data elements to their respective Clusters 
    for t in range(len(tt)):
        for i in range(len(labels)):
            if labels[i] == tt[t]:
                kk[t]+=[i]

    linkage_data = linkage(input_data, method='ward', metric='euclidean')
    complete_clustering = linkage(input_data, method="complete", metric="euclidean")
    average_clustering = linkage(input_data, method="average", metric="euclidean")
    single_clustering = linkage(input_data, method="single", metric="euclidean")

    if STATUS == 2:
        figure(figsize=(20, 12), dpi=200)
        plt.tight_layout()
        plt.subplot(2,2,1)
        dendrogram(linkage_data)
        plt.title("Automated Clustering",fontsize = 10)
        plt.xlabel("Sample Number in Respective Clusters")
        plt.ylabel("Distance b/w each Sample (in respective data scale)")
        #plt.savefig("Automated_Clustering.png")


        plt.subplot(2,2,2)
        dendrogram(single_clustering)
        plt.title("Single Clustering")
        plt.xlabel("Sample Number in Respective Clusters")
        plt.ylabel("Distance b/w each Sample (in respective data scale)")
        #plt.savefig("Simple_Clustering.png")


        plt.subplot(2,2,3)
        dendrogram(complete_clustering)
        plt.title("Complete Clustering")
        plt.xlabel("Sample Number in Respective Clusters")
        plt.ylabel("Distance b/w each Sample (in respective data scale)")     

        plt.subplot(2,2,4)
        dendrogram(average_clustering)
        plt.title("Average Clustering")
        plt.xlabel("Sample Number in Respective Clusters")
        plt.ylabel("Distance b/w each Sample (in respective data scale)")
        plt.savefig("Hierarchical_clustering.png")
       
        plt.show()
        
    elif STATUS == 1:
        figure(figsize=(20, 12), dpi=100)
        dendrogram(linkage_data)
        plt.title("Automated Clustering",fontsize = 10)
        plt.xlabel("Sample Number in Respective Clusters")
        plt.ylabel("Distance b/w each Sample (in respective data scale)")
        plt.savefig("Automated_Clustering.png")
        plt.show()
    
    # Returning a Dictionary with The Number of Clusters and respective Elements within
    return kk, labels
