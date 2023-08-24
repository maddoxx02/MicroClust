# This Function is used to read the raw Data from the Folder; 
# The scan Data is stored within a Folder & the Path of this folder is given as input.

import os
import numpy as np
import pandas as pd
np.random.seed(1000)

def reader(FILE_PATH): # Input is given as the Path of the Folder Where all Data is stored. 
    
    # Change Directory of the Kernel
    os.chdir(FILE_PATH) #os.getcwd()
    
    # Storing the Path of each file within the Folder
    file_paths = []    
        
    # Checking if all files are TXT or not, Read only TXT files
    for file in os.listdir(): 
        if file.endswith(".txt"):
            file_paths.append(f"{FILE_PATH}\{file}")   

    # A Dictionary to store the Path of each File with an ID Number (assigned from 0 - max)
    a = {}  
    k = 0
    for ele in file_paths:
        a[k] = ele
        k+=1
    
    # A Dictionary to store the DATA
    b = {}   
    for i in range(len(a)):
        b[i] = pd.read_csv(a[i], delimiter = "\t", header = None) # Delimiter can be added as a feature or manually changed when required.
    
    # Taking the Dimension of the first file. (It is assumed that the Files will have same shape- has to be modded) # What happens if Data of Different Dimensions are give...? Something to be done in the future.
    D1, D2 = b[0].shape 
    
    # Creating a variable suitable for input to the Clustering Algorithm 
    X = np.random.rand(len(b), D1*D2) # (Number of elements, Dimension in 1D)
    for j in range(len(X)):
        X[j] = b[j].to_numpy().reshape(1,-1)
    
    #Deletion to free up space
    #del a
    
    # returns the RESHAPED Data to be fed to the Clustering Algorithm & the Original Data for reference 
    return X, b, a