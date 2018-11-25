import numpy as np
import matplotlib.pyplot as plt
import spectral

import time
import os
import sys

#Function found on stackoverflow
def printProgressBar (iteration, total, prefix = 'Progress: ', suffix = ' Complete', decimals = 1, length = 40, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total: 
        print()

        
def print_shape(**kwargs):
    """
    Print multiple shapes of np.ndarray
    """
    for key, value in kwargs.items():
        print ("%s: %s" %(key, value.shape))
        
        
def displayPrincipalComponents(X, cmap="gray"):
    """
    Display principal components of the sensor data array
    ...
    
    Parameters
    ----------
    X : np.ndarray of dim MxNxP
        Sensor data of MxN pixels and P bands
    cmap : str, optional
        Custom color map for matplotlib
    """
    pc = spectral.principal_components(X_train)
    plt.figure()
    plt.imshow(pc.cov, cmap=cmap)
    
    return pc


def displayImage(X, img_num=3, cmap="gray"):
    """
    Display image from sensor data array
    ...
    
    Parameters
    ----------
    X : np.ndarray of dim MxNxP
        Sensor data of MxN pixels and P bands
    img_num : int, optional
        Display band 'img_num'
    cmap : str, optional
        Custom color map for matplotlib
    """
    plt.figure()
    plt.imshow(X[:,:,img_num], cmap=cmap)
    