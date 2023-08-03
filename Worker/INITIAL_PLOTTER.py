# This Function performs PLotting of the Raw Data (Sample Images without Clustering) i.e. just for reference

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from skimage.io import imread, imshow

# Presets for Plots 
sns.set(style="darkgrid")
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100


# The function takes 1 field of input, 
# 1. Original = the original data as attained from Reader.py

def plotter(original):
          
    rows = 8 #has to be modded
    columns = 8
    fig = plt.figure(figsize=(100, 95), dpi = 50) # Size & DPI of Plot
    
    # Generation of Samples plots
    for i in range(len(original)):    
        
        fig.add_subplot(rows,columns, i+1)
        
        plt.imshow(original[i], aspect = "equal", cmap = "gray")
        plt.axis('off')
        plt.title("Sample Number = "+str(i),fontsize= '60')
        
        plt.tight_layout()
