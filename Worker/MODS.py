import pandas as pd
import numpy as np
from statistics import mean
from numpy.fft import rfft
import cmath as cmath

# one-dimensional discrete Fourier Transform 

def logger(INPUT):
    
    value = []
    
    CC = INPUT.copy()
    
    for i in range(len(INPUT)):
        value = []
        for j in range(len(INPUT[i])):
            if INPUT[i][j] == 0:
                value.append(0)
            else:
                value.append((cmath.log10(INPUT[i][j])).real)
    
        CC[i] = value
    
    return CC

# Direct FFT

# This function calculates the Fourier Transform of the Data & returns they individual Columns & Rows. The Input is the source data in the form of a pandas table

def mag_maker(Input):
    
    # columns     
    cols = []
    for i in range(len(Input)):
        cols.append(rfft(Input[i]).real)
    
    # Rows
    rows = []
    for j in range(len(Input)):
        rows.append(rfft(Input.iloc[j]).real)
    
    logged_cols = pd.DataFrame(logger(cols)).transpose()
    cols_l = []
    for i in range(len(logged_cols)):
        cols_l.append(mean(logged_cols.iloc[i]))
        
    logged_rows = pd.DataFrame(logger(rows))
    rows_l = []
    for i in range(len(logged_rows.axes[1])):
        rows_l.append(mean(logged_rows[i]))
        
    return cols_l, rows_l



def Process_1D_F(inn):

    trial = mag_maker(inn[0])

    cols = np.random.rand(len(inn), len(trial[0]))
    rows = np.random.rand(len(inn), len(trial[0]))

    for i in range(len(inn)):
        cols[i], rows[i] = mag_maker(inn[i])
        
    return cols, rows

        

# This function returns the fourier transform of the sample categorized by Columns & rows in 1Dimensional arrays
















import pandas as pd
import numpy as np
from statistics import mean


from numpy.fft import rfft
from numpy.fft import rfft, rfftfreq, irfft, rfft2, fft

import cmath as cmath
import cv2

# Two-Dimensional Discrete Fourier transform 


# Performing Fourier Tranform on the Source data as a 2D entity 
def data_class(IMAGE_DATA): 
    
    f = np.log(np.abs(np.fft.fft2(IMAGE_DATA)))
    
    return f
    

def Process_2D_F(INPUT):

    value = np.random.rand(len(INPUT), len(INPUT[0])*len(INPUT[0]))

    CC = INPUT.copy()
    
    for i in range(len(INPUT)):
        value[i] = (data_class(INPUT[i])).reshape(1,-1)
    

    return value



# Alternatively there is a method to use the image as source input and then perform operations on it. During testing
# i tried using Images as input for clustering,by using images as input, performing Fourier Transform & then clustering
# using kmeans, the clusters got classified as per its Size

def image_class(IMAGE):
    
    img = cv2.imread(IMAGE,0)
    
    i_f = np.log(np.abs(np.fft.fft2(img)))
    
    return i_f
    
    
    
    