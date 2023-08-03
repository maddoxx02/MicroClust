import os

def address(FILE_PATH):
    
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
        
    return a