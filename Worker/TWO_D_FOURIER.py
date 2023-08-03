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
    
    