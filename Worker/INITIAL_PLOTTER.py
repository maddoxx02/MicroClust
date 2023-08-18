# This Function performs PLotting of the Raw Data (Sample Images without Clustering) i.e. just for reference

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from skimage.io import imread, imshow
import math

# Presets for Plots 
sns.set(style="darkgrid")
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100


# The function takes 1 field of input, 
# 1. Original = the original data as attained from Reader.py

def plotter(original):
          
    rows, columns = col_calc(original)

    fig = plt.figure(figsize=(100, 95), dpi = 100) # Size & DPI of Plot
    
    # Generation of Samples plots
    for i in range(len(original)):    
        
        fig.add_subplot(rows, columns, i+1)
        
        plt.imshow(original[i], aspect = "equal", cmap = "gray")
        plt.axis('off')
        plt.title("Sample Number = "+str(i+1),fontsize= '60')
        plt.tight_layout()
        plt.savefig("Initial Data", bbox_inches='tight')


def col_calc(input):

    f_r = math.ceil(len(input)/10)
    f_c = math.ceil(len(input)/10)
    target = len(input)

    while target >= f_r*f_c:
        f_c = f_c+1

    return f_r,f_c    