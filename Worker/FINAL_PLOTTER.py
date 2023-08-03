# This Function performs PLotting of the Generated Clusters

import seaborn as sns
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
from skimage.io import imread, imshow

# Presets for Plots 
sns.set(style="darkgrid")
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100


# The function takes 2 fields of input, 
# 1. Data = The Results of Clustering in the format of A dictionary with Lists
# 2. Original = the original data as attained from Reader.py

def plotter_2(data, original):
    
    for i in range(len(data)):
        
        splitter(data, original, i)

        
def splitter(data, original, ctr_1):
    
    fig = plt.figure(figsize=(25, len(data[ctr_1])*7), dpi = 100) # Size & DPI of Plot
    fig.suptitle("\n"+"Cluster "+str(ctr_1+1), fontsize=60, y = 1) # Cluster Numbering
   
    # Orientation of the Plots within the Figure
    rows = len(data[ctr_1]) 
    columns = 5
    
    for j in range(len(data[ctr_1])):   
        fig.add_subplot(rows, columns, j+1) # Adding Plots within 
                
        plt.imshow(original[data[ctr_1][j]], aspect = "equal", cmap = "gray")
        plt.axis('off')
        plt.title("Sample = "+str((data[ctr_1][j])+1),fontsize = 30)
        plt.tight_layout()
        plt.savefig("Cluster "+str(ctr_1+1), bbox_inches='tight')
    
        
