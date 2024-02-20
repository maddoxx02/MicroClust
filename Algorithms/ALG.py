# A_1 = K_MEANS
# A_2 = K_MEANS_PLUS
# A_3 = K_MEANS_BISECT
# A_4 = DBSCAN_AUTO
# A_5 = DBSCAN_MANUAL
# A_6 = HDBSCAN
# A_7 = HIERARCHY
# A_8 = FUZZY_C
# A_9 = MEAN_SHIFT
# A_10 = AFFINITY
# A_11 = BIRCH
# A_12 = OPTICS
# A_13 = SPECTRAL
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from skimage.io import imread, imshow

# Presets for Plots 
sns.set(style="darkgrid")
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100


import numpy as np 
np.random.seed(0) 

#------------------------------------------------------------------------------------------------------------------------------------------

# This function performs Clustering on the Data; 

from sklearn.cluster import KMeans


def K_MEANS(input_data, cluster):  

    # K_Means algorithm
    Kmean = KMeans(n_clusters = cluster, random_state=0, init = 'random')#, n_init = 100)
    
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


#------------------------------------------------------------------------------------------------------------------------------------------

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
#------------------------------------------------------------------------------------------------------------------------------------------
# Bisect K means

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
#------------------------------------------------------------------------------------------------------------------------------------------

# Determining optimal EPS value: https://iopscience.iop.org/article/10.1088/1755-1315/31/1/012012/pdf
# Distance can be dependent on Elements or chemicals used for the polymer

# Minimal domain knowledge
#pip install kneed

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
from kneed import KneeLocator

def DBSCAN_AUTO(input_data, neigh, min_samp):

    neigh = NearestNeighbors(n_neighbors=neigh) #INPUT_1

    nbrs = neigh.fit(input_data)

    distances, indices = nbrs.kneighbors(input_data)

    distances = np.sort(distances, axis=0)
    distances = distances[:,1]

    x = range(0, len(distances))

    kn = KneeLocator(x, distances, curve='convex', direction='increasing')
    
    clustering = DBSCAN(eps = distances[kn.knee], min_samples = min_samp, algorithm='brute').fit(input_data)#random_state=0 # MIN_SAMPLES

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

#------------------------------------------------------------------------------------------------------------------------------------------

# DBSCAN Manual knee locater
from sklearn.cluster import DBSCAN
import numpy as np

def DBSCAN_MANUAL(input_data, eps, min_samp):
    
    clustering = DBSCAN(eps=eps, min_samples=min_samp)# min samples = minimum samples to be considered as a Centroid
    
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

    # Returning a Dictionary with The Number of Clusters and respective Elem
    return kk, clustering.labels_
#------------------------------------------------------------------------------------------------------------------------------------------
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
#------------------------------------------------------------------------------------------------------------------------------------------
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
#------------------------------------------------------------------------------------------------------------------------------------------
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
#------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.cluster import MeanShift


def MEAN_SHIFT(input_data, bw):

    clustering = MeanShift(bandwidth = bw) #486

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


#------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.cluster import AffinityPropagation
def AFFINITY(input_data, damper):
    
    clustering_AFP = AffinityPropagation(damping = damper, random_state=0, max_iter=300, convergence_iter=54)#random_state=5
    
    clustering_AFP.fit(input_data)
    
        # A Dictionary to Store the Cluster IDs & the Respective Data with IDs
    kk = {}
    temp = []

    # Creating the Key for the number of Clusters as mentioned
    for k in range(len(np.unique(clustering_AFP.labels_))):
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

#------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.cluster import Birch
def BIRCH(input_data, clusters,thres, bf):#, threshold, clusters):

    clustering = Birch(threshold = thres, n_clusters=clusters, branching_factor=bf)

    clustering.fit(input_data)
    
    # A Dictionary to Store the Cluster IDs & the Respective Data with IDs
    kk = {}
    temp = []

    # Creating the Key for the number of Clusters as mentioned
    for k in range(clusters):
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


#------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.cluster import OPTICS

def OPTICS_AUTO(input_data, min_samp): #max_eps = 475
    
    clustering = OPTICS(min_samples = min_samp,metric='euclidean', algorithm='brute')

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
#------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.cluster import SpectralClustering
import numpy as np


def SPECTRAL(input_data, clusters):

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
#------------------------------------------------------------------------------------------------------------------------------------------
