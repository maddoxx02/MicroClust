from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics import davies_bouldin_score
import pandas as pd


def Metric_2(data, labels, Algo):

    A = metrics.silhouette_score(data, labels, metric='euclidean') 
    # Distance b/w inter & intra samples per cluster per sample # Higher the better -1 is very bad, +1 is the best
    
    B = metrics.calinski_harabasz_score(data, labels)# (Variance Ratio) # Higher the better, no limit 
    
    C = davies_bouldin_score(data, labels) #(AVG SIMIALRITY WITHIN CLUSTERS) minimum score is 0.1 (lower the better)
    
    #print("Silhouette Score for ",len(np.unique(labels)), "is =", A)
    #print("Calinski Harabasz Score for ",len(np.unique(labels)), "is =", B)
    #print("Davies Bouldin Score for ",len(np.unique(labels)), "is =", C)
    
    COMBINED_SET_2 = list(["{0:.2f}".format(A), "{0:.2f}".format(B), "{0:.2f}".format(C)])
    
    S_2 = pd.DataFrame({Algo:COMBINED_SET_2}).transpose()
    S_2.columns = ["Silhouette_Score", "Calinski_Harabasz_Score", "Davies_Bouldin_Score"]
    
    return COMBINED_SET_2, S_2