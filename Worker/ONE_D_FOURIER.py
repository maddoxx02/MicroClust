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