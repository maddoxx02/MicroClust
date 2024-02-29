# This code can be used to filter the Atomic Force Microscopy (AFM) scan data from Gwyddion format of (.txt) to (.csv). 
# The code creates a CSV folder and stores all the files in (.csv) format with the same nam. 
# The code also removes the first 4 lines of metadata present after exporting from Gwyddion software 

# Input = Directory of Files where (.txt) AFM scan data is stored 
# Output = None 

import os
import pandas as pd


def TXT2CSV(txt_folder):

    os.chdir(txt_folder)

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

    Temp_path = txt_folder

    for i in range(len(a)):

        if i == 0: 
            os.mkdir('CSV')
            df1 = pd.read_csv(a[i], header = None)
            df2 = df1.drop([0,1,2,3]).reset_index(drop = True)
            df2.to_csv('TEMP_DATA.txt', index=False, header=False)
            df3 = pd.read_csv('TEMP_DATA.txt', delimiter='\t', header = None)
            os.remove("TEMP_DATA.txt")
            df3.to_csv(txt_folder+'\\CSV'+ a[i].replace(txt_folder, '').replace('.txt','.csv'), header = False, index = False)
        else:

            df1 = pd.read_csv(a[i], header = None)
            df2 = df1.drop([0,1,2,3]).reset_index(drop = True)
            df2.to_csv('TEMP_DATA.txt', index=False, header=False)
            df3 = pd.read_csv('TEMP_DATA.txt', delimiter='\t', header = None)
            os.remove("TEMP_DATA.txt")
            df3.to_csv(txt_folder+'\\CSV'+ a[i].replace(txt_folder, '').replace('.txt','.csv'), header = False, index = False)


    if i == len(a):
        print("Completed processing TXT files to CSV")



