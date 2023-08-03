import pandas as pd 
from sklearn import metrics

def Metric_1(Truth, Predi, Algo):
    
    rand_index = metrics.rand_score(Truth, Predi)*100
    adjusted_rand_index = metrics.adjusted_rand_score(Truth, Predi)*100
    MMI = metrics.adjusted_mutual_info_score(Truth, Predi)  
    Hom = metrics.homogeneity_score(Truth, Predi)
    Comp = metrics.completeness_score(Truth, Predi)
    VV = metrics.v_measure_score(Truth, Predi)
    FM = metrics.fowlkes_mallows_score(Truth, Predi)
    
    #print("1. Rand Index predicts the Ground Truth vs Predicted result is ", "{0:.2f}".format(rand_index), "% accurate as original groupging.")
    #print("2. Adjusted Rand Index considerds & suggests the Ground Truth & Predicted result to be ", "{0:.2f}".format(adjusted_rand_index), "% similar to the original grouping when considering the random chances a Sample is grouped into a cluster")
    #print("3. The Mutual Information is a measure of the similarity between Truth vs Predicted            : ",  "{0:.2f}".format(MMI*100))
    #print("4. The % of clusters that have its members of the same class (Homogeneity)                     : ",  "{0:.2f}".format(Hom*100))
    #print("5. The % that all types of samples of a specific type belong to a single cluster (Completeness): ",  "{0:.2f}".format(Comp*100))
    #print("6. The Harmonic Mean of Homogeneity & Completness (V Measure)                                  : ",  "{0:.2f}".format(VV*100)) # Beta stays = 1, since we do not need to deviate
    #print("7. The geometric mean of the pairwise precision and recall                                     : ",  "{0:.2f}".format(FM*100))

    COMBINED_SET_1 = list(["{0:.2f}".format(rand_index), "{0:.2f}".format(adjusted_rand_index), "{0:.2f}".format(MMI*100), "{0:.2f}".format(Hom*100),
                         "{0:.2f}".format(Comp*100), "{0:.2f}".format(VV*100), "{0:.2f}".format(FM*100)])
    
    S = pd.DataFrame({Algo:COMBINED_SET_1}).transpose()
    S.columns = ["Rand_Index", "Adj_Rand_Index", "MMI", "Homogeneity", "Completness", "V Measure", "Geometric Mean"]
    
    #TEST_T.index = ["K Means", "K Means++", "K Means Bisect", "Hierarchy", "Fuzzy C", "BIRCH", "SPECTRAL"]
    
    return COMBINED_SET_1, S
    
    
    
    
    
    
    
    
    
    
def PRF(Truth, Predi):


    S = pd.DataFrame({"Cluster":np.unique(Truth),"TP":[0,0,0],"FN":[0,0,0],"FP":[0,0,0],"TN":[0,0,0]}, index = list(np.unique(Truth)))

    FULL = pd.DataFrame({"Cluster":np.unique(Truth),"Individual Precision":[0,0,0],"Individual Recall":[0,0,0],"Individual F-Measure":[0,0,0]}, index = list(np.unique(Truth)))

    if len(Truth) != len(Predi):
        print("Not Equal Sizes of Input")
    else:
        print("There are", len(np.unique(Truth)), "clusters present")



    elements = np.unique(Truth)

    for ele in (elements):

        TP = 0
        FN = 0
        FP = 0
        TN = 0

        for i in range(len(Truth)):

                if Truth[i] == ele and Truth[i] == Predi[i]:
                    TP+=1

                if Truth[i] == ele and Truth[i] != Predi[i]:
                    FN+=1

                if Truth[i] != ele and Truth[i] != Predi[i]:
                    FP+=1

                if Truth[i] != ele and Truth[i] == Predi[i]:
                    TN+=1

            #print("Situation =", elements[c], Truth[i], Predi[i])


        S.loc[S["Cluster"] == ele, ["TP"]] = TP
        S.loc[S["Cluster"] == ele, ["FN"]] = FN
        S.loc[S["Cluster"] == ele, ["FP"]] = FP
        S.loc[S["Cluster"] == ele, ["TN"]] = TN



        FULL.loc[FULL["Cluster"] == ele, ["Individual Precision"]] = (S.loc[S["Cluster"] == ele]["TP"])/(S.loc[S["Cluster"] == ele]["TP"]+S.loc[S["Cluster"] == ele]["FP"])
        FULL.loc[FULL["Cluster"] == ele, ["Individual Recall"]] = (S.loc[S["Cluster"] == ele]["TP"])/(S.loc[S["Cluster"] == ele]["TP"]+S.loc[S["Cluster"] == ele]["FN"])
        FULL.loc[FULL["Cluster"] == ele, ["Individual F-Measure"]] = 2*(FULL.loc[FULL["Cluster"] == ele]["Individual Precision"]*FULL.loc[FULL["Cluster"] == ele]["Individual Recall"])/(FULL.loc[FULL["Cluster"] == ele]["Individual Precision"]+FULL.loc[FULL["Cluster"] == ele]["Individual Recall"])
    
        res = pd.DataFrame({"Cluster": ["Overall"],
                    "Individual Precision":mean(FULL["Individual Precision"]),
                   "Individual Recall": mean(FULL["Individual Recall"]),
                   "Individual F-Measure":mean(FULL["Individual F-Measure"])})
    
        Final_res = pd.concat([FULL,res],ignore_index = True)
        
    print(S)
    print("\n", Final_res)
    Final = list([ "{0:.2f}".format(mean(FULL["Individual Precision"])*100),  
                  "{0:.2f}".format(mean(FULL["Individual Recall"])*100),
                   "{0:.2f}".format(mean(FULL["Individual F-Measure"]*100))])
    return S, Final_res, Final
