# MicroClust


! BADGES



### *MicroClust V1.0* is a python based toolbox that performs classificaiton of small scale surface images using Unsupervised clustering algorithims

Images from Atomic force microsope (AFM), scanning electorn microsope (SEM) etc, can be classifed using 12 of the most widely used clustering algorithims and further validated using Ground truth and self-evaluatory metrics. The toolbox can perform a 1D and 2D Fourier transform on the image data prior to classification. 

![MicroClust](Toolkits/MicroClust_Compressed.png)


#### Algorithims available are: 
 1. K Means 
 2. K Means ++ 
 3. K Means Bisect
 4. Hierarchy 
 5. Fuzzy C Means
 6. Spectral 
 7. DBSCAN (Automated & Manually tuned) 
 8. HDBSCAN 
 9. Mean Shift
 10. OPTICS
 11. Affinity Propagation
 12. BIRCH 

#### Metrics available: 
Ground Truth evaluation: 
 1. Rand Index
 2. Adjusted Rand Index
 3. Adjusted Mutual Information Score
 4. Homogeneity
 5. Completeness
 6. V Measure
 7. Fowlkes Mallows Score
    
Self-Evaluation:
 1. Silhouette Score
 2. Calinski Harabasz Index
 3. Davies Bouldin Index


The **V1.0** version is a first level iteration of the toolbox. Here, the use
 
 # Simulation Results

simulations are in the folder


# How to use the toolbox.?

``` clone git repo ```



# Citing the toolbox:
If you use this toolbox, pLease cite using: 



## Requirements 
This toolbox requires the following libraries to run: 
 1. Sklearn
 2. [Fuzzy C Means](https://github.com/omadson/fuzzy-c-means)

