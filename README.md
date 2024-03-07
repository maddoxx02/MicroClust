# MicroClust


![GitHub last commit](https://img.shields.io/github/last-commit/maddoxx02/MicroClust)
![GitHub Release](https://img.shields.io/github/v/release/maddoxx02/MicroClust)
![GitHub License](https://img.shields.io/github/license/maddoxx02/MicroClust)




### *MicroClust V1.0* is a python based toolbox that performs classification of small scale surface images using Unsupervised clustering algorithims

Images from an atomic force microsope (AFM), a scanning electorn microsope (SEM) or similar imaging modalities, can be classifed using 12 of the most widely used clustering algorithms which can be further validated using Ground truth and self-evaluatory metrics. The toolbox can perform a 1D and 2D Fourier transform on the image data prior to classification. 

![MicroClust](Add-ons/Repo_Images/IMG_MicroClust_Compressed.png)


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


The **V1.0** version is a first iteration of the toolbox. Here, the input provided from the user is the ```path``` to the directory of images to be classifed. The output produced is a visualization of the clusters generated and the generated scores of each metric. 
Algorithims 1 to 6 are tuned implicitly (internally) and only require the expected number of clusters to be provided, where as algorithms 7 - 12 are explicitly tuned (tuned by user's input) and require more information to be provided such as the minimum number of neighours to be considered a centroid point etc. 

This toolbox is described in the publication ***"Benchmarking Unsupervised Clustering Algorithms for Atomic Force Microscopy Data on Polyhydroxyalkanoate Films"***


 # Simulation Results

The experiments & results from the work [LINK TO PAPER](LINK TO PAPER) is available in the ```Simulations``` folder as Python Notebooks. The algorithms have already been tuned to the respective data and perform as in the article. The dataset used for simulations is available at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10649355.svg)](https://doi.org/10.5281/zenodo.10649355)

An overview of the simulations performed is shown in the figure below:

![Overview](Add-ons/Repo_Images/IMG_Outline.jpg)


The following image visualizes the transformation performed within the toolkit to feed data to the algorithms

![Overview](Add-ons/Repo_Images/IMG_Transformation.png)


The ```Simulations``` directory consists of three cases of experiments where specific features of the AFM data were being sought. 
 - Case 1 - Classification of the AFM data as per ***scan size***
 - Case 2 - Classification of the AFM data as per ***polymer type***
 - Case 3 - Classificaiton of the AFM data as per ***thickness of films***

Under each case, the simulations are further divided into specific sub-cases (A,B,C,D) based on the pool of data used for classification. The contents of the folders include: 
 - The visualizations of the initial scans & their classified results
 - Benchmarked scores of all metrics
 - Graphical visualization of Hierarchial relation


Further, usage & features of the toolkits are explained in the sections below. 


# How to use the toolbox

Instructions to ***1.0V*** of the toolkit:

- *Installation:*
```
git clone https://github.com/maddoxx02/MicroClust
```
``` git clone https://github.com/maddoxx02/MicroClust <YOUR CUSTOM DIRECTORY>```
(Future, versions will have ```pip install```)
*For dependencies scroll to the end of the readme.*


- *Import & usage:*
```
import sys
sys.path.insert(1,'Y:\MicroClust')
```
```sys.path.insert(1, <DIRECTORY OF MICROCLUST INSTALLATION>)```


- *Loading Data*
To load data into your workspace, use:
```
import Worker.DATA_READER as DR
```
```
data, original, adrs = DR.reader(<DIRECTORY OF AFM DATA>)
```
where, 
```data``` stores the data in the vectorized format as shown in the previous section. The raw data is used computation and metric calculation. 
```original``` is used to create visualizations of the data fed (before and after classification).
```adrs``` is a dictionary to store the individual ```path``` of each file being used (as a reference).

- *Peforming 1D and 2D Fourier Transforms*
```
One_Data = DM.Process_1D_F(original)
Two_Data = DM.Process_2D_F(original) 
```

- *Plotting Initial set of data*
```
import Worker.INITIAL_PLOTTER as IPLOT         
import Worker.FINAL_PLOTTER as FPLOT           
```
where, 
```IPLOT.plotter(original)``` is used to create a single visualization of all images provided before classification
```FPLOT.plotter(list(clustered labels), original)``` creates images of each cluster with elements within


- Performing Operations
 1. ***Algorithms***

To import algorithms, intialize: 
```
import Algorithms.ALG as ALG 
```

Each algorithm in the toolkit can be called individually, to improve readability & ease of access the algorithms are split into two groups:

 a. ***Implicitly Tuned*** (internally tuned algorithms) - These algorithms require the final number of clusters to be provided as input
```
ALG.K_MEANS(data, <NUM_CLUSTERS>) 
ALG.K_MEANS_PLUS(data, <NUM_CLUSTERS>)
ALG.K_MEANS_BISECT(data, <NUM_CLUSTERS>)
ALG.FUZZY_C(data, <NUM_CLUSTERS>)
ALG.SPECTRAL(data, <NUM_CLUSTERS>)
ALG.HIERARCHY(data, ALG.HIERARCHY(data, <NUM_CLUSTERS>, <GRAPH>)  
```
Where, 
- ```K Means, K Means ++ & K Means Bisect``` is set to have maximum iterations of ***300*** with ***10*** initializations, while K Means bisect follows the ***Greedy approach*** for cluster centre initialization and a bisecting strategy of ***Biggest Inertia***

- ```Fuzzy C Means``` has a degree of Fuzziness ***2*** and maximum iterations of ***300***
  
- ```Spectral algorithm``` utilizes ***arpack*** decomposition strategy with ***10*** initializations and ***Radial basis function*** to calculate affinity
  
- In hierarchy algorithm uses an ***Euclidean affinity metric*** and a ***ward*** linkage criteria , ```<GRAPH>``` can be either ```1``` or ```0```, to produce a dendogram graph of the clustered results or not. 



 b. ***Explicitly Tuned*** (manually tuned algorithms) - These algorithms require additional hyperparameters to be tuned by the use before usage
```
T0_A7 = ALG.DBSCAN_AUTO(data, <NEAREST NEIGHBOURS THRESHOLD>, <MINIMUM NUM OF SAMPLES>)
T0_A8 = ALG.DBSCAN_MANUAL(data, <EPSILON>, <MINIMUM NUM OF SAMPLES>)      
T0_A9 = ALG.HDBSCAN_MANUAL(data, <MINIMUM CLUSTER SIZE>, <MINIMUM NUM OF SAMPLES>, <MAXIMUM CLUSTER SIZE>)
T0_A10 = ALG.MEAN_SHIFT(data, <BANDWIDTH>)
T0_A11 = ALG.OPTICS_AUTO(data, <MINIMUM NUM OF SAMPLES>)     
T0_A12 = ALG.AFFINITY(data,<DAMPING FACTOR>)
T0_A13 = ALG.BIRCH(TWO_Data, <NUM_CLUSTERS>, <THRESHOLD>, <BRANCHING FACTOR>)

```
Where, 
- ```DBSCAN (Automated)``` performs calculation of the knee point convergence internally. The aglorithm uses ```Brute Force``` algorithm to calculate the distance of nearest neighbours
  - ***NEAREST NEIGHBOURS THRESHOLD*** - The minimum distance between neighours possible during clustering in dimension space
  - ***MINIMUM NUM OF SAMPLES*** - The minimum number of samples in the neighbourhood to be considered a core point (centroid for a cluster)
  
- ```DBSCAN (Manual)``` the knee point is determined using the ```EPSILON``` value provided. The algorithm used to calculate the distance of nearest neighbours is ```Nearest neighbours```
  - ***EPSILON*** - The Maximum distance between two samples of one to be considered in the neighbour hood of the other
  - ***MINIMUM NUM OF SAMPLES*** - The minimum number of samples in the neighbourhood to be considered a core point (centroid for a cluster)
 
- ```HDBSCAN``` uses the ```brute force``` approach to compute core distances with a cluster selection method using ```excess mass``` approach
  - ***MINIMUM CLUSTER SIZE*** - Minimum number of samples in a group to be considered a cluster. Clusters smaller than this is considered as noise
  - ***MINIMUM NUM OF SAMPLES*** - The minimum number of samples in the neighbourhood to be considered a core point (centroid for a cluster)
  - ***MAXIMUM CLUSTER SIZE*** - The maximum possible size of a cluster in the dimension space after which another cluster is created
    
- ```Mean Shift``` the algorithm is set to perform ***300*** iterations to converge
  - ***Bandwidth*** - Defines the scale (size of the window) of mean to be calculated for the given data points
  
- ```OPTICS``` uses the ***Euclidean*** metric to calculate the distance between samples and a ```brute force``` approach to compute nearest neighbours 
  - ***MINIMUM NUM OF SAMPLES*** - The minimum number of samples in the neighbourhood to be considered a core point (centroid for a cluster)

- ```Affinity Propagation``` performs ```300``` iterations to converge using a ```Eculidean``` affinity metric and ***54*** a threshold for convergence
  - ***DAMPING FACTOR*** - The coefficent of weightage of messages exchanged between data points to create clusters (i.e. the damping effect on each message after reaching a point & returning during clustering cycle)

- ```BIRCH```
  - ***THRESHOLD*** - The distance between closest subclusters above which the subclusters are merged into a single cluster
  - ***BRANCHING FACTOR*** -  The maximuim number oif clustering feature trees under each node, above which subclusters are created



The algorithms return two arrays: 
1. A Dictionary of the predicted labels
2. The predicted label set (as a numpy array)
  

- ***Performing Operations on the data***
  
  Include
  ```
  import Worker.MODS as DM
  ``` 
1. 1D Fourier Transform
```
One_D = DM.Process_1D_F(original) 
```

2. 2D Fourier Transform
```
Two_D = DM.Process_2D_F(original)
```

This data can be fed to the algorithm as follows:
```
ALG.SPECTRAL(One_D[1], 3)
ALG.MEAN_SHIFT(Two_D, 490)
```
where,
```One_D``` has transformed data in the orientation of ```[0] = columns``` and ```[1] = rows```




 4. ***Metrics***
To use metrics initalize:
```
import Metrics.GroundTruth_Metrics as MC_1     
import Metrics.SelfEvaluatory_Metrics as MC_2  

```

 To use Ground Truth metrics, the truth label set has to be intialized as a numpy array e.g.
 ```
 GD = np.array([0,0,0,1,1,1])
 ```

 Calling a group of metrics. The input for the ground truth metrics is to be given as:  
 ```
 M1 =  MC_1.Metric_1(<GROUND TRUTH>, <RESULT OF ALG[1]>,<NAME OF ALG>)
 ```
 Self Evaluatory metrics are called as: 
 ```
 M2 =  MC_2.Metric_2(<RESULT OF ALG[1]>,<NAME OF ALG>)
 ```

e.g. 

```
GD = np.array([0,0,1,2])
Test_1 = ALG.K_MEANS(data, 3)
Result_G = MC_1.Metric_1(GD, Test_1[1], "K means")
Result_S = MC_2.Metric_2(Test_1[1], "K means")

```

it returns a single dataframe slice which can be exported or saved as requried. 


5. ***Visualizing Results***
```
FPLOT.plotter_2(Test_1[0], original)
```
This produces the visualizastion of each cluster & its elements together. 


## Additional tools

Within MicroClust, there are two scripts to perform conversion operations from data exported from Gwyddion (.txt) to MicroClust compatible format (.csv). These scripts can be used on the dataset mentioned earlier. 

 - ***TXT2CSV.py*** : this script is used to convert Gwyddion format of data ***(.txt)*** to ***(.csv)*** (the dimension of the file remains unchanged)
   The script takes the ```path``` of the directory containing the (.txt) files as input and creates a folder named ```CSV``` within which all (.csv) files are stored. 
 
 - ***CSV2IMG.py*** : this script is used to export ***(.csv)*** format of data to ***(.png)*** images (of the surface)
   The script takes the ```path``` of the directory containing the (.csv) files as input and creates a folder named ```IMG``` within which all (.png) images are stored. 


```
TXT2CSV(<PATH TO DIRECTORY>)
CSV2IMG(<PATH TO DIRECTORY>)
```


# Citing the toolbox:
If you use this toolbox, please cite the following paper: 



## Requirements 
This toolbox requires the following libraries to run: 
 1. Sklearn (1.3.0)
 2. [Fuzzy C Means](https://github.com/omadson/fuzzy-c-means)(1.7.0)
 3. Numpy (1.22.4)
 4. Seaborn (0.11.2)
 5. skimage (0.19.2)
 6. matplotlib (3.5.2)
 7. kneed (0.8.2)

