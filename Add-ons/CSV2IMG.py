# This code can be used to convert (.csv) files to (.png) images the proccessed Atomic Force Microscopy (AFM) scan data from (.csv). 
# The code creates a IMG folder and stores all the files in (.png) format with the same name. 

# Input = Directory of Files where processed (.csv) AFM scan data are stored 
# Output = None 

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 100

def CSV2IMG(csv_folder):

    os.chdir(csv_folder)

    file_paths = []

    # Checking if all files are TXT or not, Read only TXT files
    for file in os.listdir(): 
        if file.endswith(".csv"):
            file_paths.append(f"{csv_folder}\{file}")   


    # A Dictionary to store the Path of each File with an ID Number (assigned from 0 - max)
    a = {}  
    k = 0
    for ele in file_paths:
        a[k] = ele
        k+=1

    Temp_path = csv_folder

    for i in range(len(a)):

        if i == 0: 
            os.mkdir('IMG')
            df1 = pd.read_csv(a[i])
            plt.axis('off')
            plt.imshow(df1, aspect = 'equal', cmap = 'gray')
            plt.savefig(csv_folder+'\\IMG'+a[i].replace(csv_folder, '').replace('.csv','.png'), aspect='equal', cmap = "gray", bbox_inches='tight',   pad_inches = 0)
            plt.show()
        else:
            df1 = pd.read_csv(a[i])
            plt.axis('off')
            plt.imshow(df1, aspect = 'equal', cmap = 'gray')
            plt.savefig(csv_folder+'\\IMG'+a[i].replace(csv_folder, '').replace('.csv','.png'), aspect='equal', cmap = "gray", bbox_inches='tight',   pad_inches = 0)
            plt.show()




