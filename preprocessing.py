import numpy as np

from sklearn.decomposition import PCA
from sklearn import preprocessing

import skimage as sk
from skimage import transform
from skimage import util

import scipy.io as sio

import matplotlib.pyplot as plt

import time
import os
import sys
import random
from random import shuffle

#our utils functions
import utils as u

def loadData():
    data_path = os.path.join(os.getcwd(),'.')
    data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
    train_labels = np.load("train_data.npy")
    test_labels = np.load("test_data.npy")
    
    return data, train_labels, test_labels

def reduceComponents(X, reduce_factor=7):
    """
    Reduce aviris sensor data array in principal components
    ...
    
    Parameters
    ----------
    X : np.ndarray of dim MxNxP
        Sensor data of MxN pixels and P bands
    reduce_factor : int, optional
        Determines the strength of dimensionality reduction
    """
    switcher = {
        1: 0.9,
        2: 0.99,
        3: 0.999,
        4: 0.9999,
        5: 0.99999,
        6: 0.999999,
        7: 0.9999999
    }
    fraction = switcher.get(reduce_factor, 7)
    pc = spectral.principal_components(X).reduce(fraction=fraction)

    # How many eigenvalues are left?

    print("Reflectance bands remaining: %s" %(len(pc.eigenvalues)))
    newX = pc.transform(X)

    #v = plt.imshow(img_pc[:,:,1], cmap="cool")
    return newX

def patch_1dim_split(X, train_data, test_data, PATCH_SIZE):
    padding = int((PATCH_SIZE - 1) / 2) #Patch de 3*3 = padding de 1 (centre + 1 de chaque coté)
    #X_padding = np.zeros(X)
    X_padding = np.pad(X, [(padding, padding), (padding, padding), (0, 0)], mode='constant')
    
    X_patch = np.zeros((X.shape[0] * X.shape[1], PATCH_SIZE, PATCH_SIZE, X.shape[2]))
    y_train_patch = np.zeros((train_data.shape[0] * train_data.shape[1]))
    y_test_patch = np.zeros((test_data.shape[0] * test_data.shape[1]))
    
    index = 0
    for i in range(0, X_padding.shape[0] - 2 * padding):
        for j in range(0, X_padding.shape[1] - 2 * padding):
            # This condition is for less frequent updates. 
            if i % 8 == 0 or index == (X_padding.shape[0] - 2 * padding) * (X_padding.shape[1] - 2 * padding) - 1:
                u.printProgressBar(index + 1, (X_padding.shape[0] - 2 * padding) * (X_padding.shape[1] - 2 * padding))
            patch = X_padding[i:i + 2 * padding + 1, j:j + 2 * padding + 1]
            X_patch[index, :, :, :] = patch
            y_train_patch[index] = train_data[i, j]
            y_test_patch[index] = test_data[i, j]
            index += 1
    
    print("\nCreating train/test arrays and removing zero labels...")
    u.printProgressBar(1, 7)
    X_train_patch = np.copy(X_patch)
    u.printProgressBar(2, 7)
    X_test_patch = np.copy(X_patch)
    
    u.printProgressBar(3, 7)
    X_train_patch = X_train_patch[y_train_patch > 0,:,:,:]
    u.printProgressBar(4, 7)
    X_test_patch = X_test_patch[y_test_patch > 0,:,:,:]
    u.printProgressBar(5, 7)
    y_train_patch = y_train_patch[y_train_patch > 0] - 1
    u.printProgressBar(6, 70)
    y_test_patch = y_test_patch[y_test_patch > 0] - 1
    u.printProgressBar(7, 7)
    print("Done.")
    
    return X_train_patch, X_test_patch, y_train_patch, y_test_patch


def dimensionalityReduction(X, numComponents=75, standardize=True):
    if standardize:
        newX = np.reshape(X, (-1, X.shape[2]))
        scaler = preprocessing.StandardScaler().fit(newX)  
        newX = scaler.transform(newX)
        X = np.reshape(newX, (X.shape[0],X.shape[1],X.shape[2]))
    
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca

def BoostDataset(X, y, n_samples=0):
    # Techniques from 
    # https://medium.com/@thimblot/data-augmentation-boost-your-image-dataset-with-few-lines-of-python-155c2dc1baec
    
    orig_shape = X.shape[0]
    index = orig_shape
    print("Boosting Dataset...")
    for i in range(n_samples):
        if i % 5 == 0 or i + 1 == n_samples:
            u.printProgressBar(i + 1, n_samples)
        num_sample = random.randint(0, orig_shape)
        patch = X[num_sample,:,:,:]
        #print(patch.shape)
        num = random.randint(0, 4)
        if (num == 0):
            new_patch = np.flipud(patch)
            
        if (num == 1):
            new_patch = np.fliplr(patch)
            
        if (num == 2):
            new_patch = sk.util.random_noise(patch)
            
        if (num == 3 or num == 4):
            random_degree = random.uniform(-25, 25)
            new_patch = sk.transform.rotate(patch, random_degree)
            
        #print(new_patch.shape)
        #time.sleep(5)
            
        X = np.append(X, [new_patch], axis=0)
        y = np.append(y, y[num_sample])
    
    return X, y

# @TODO: Vérifier la validité 
def oversampleWeakClasses(X, y):
    uniqueLabels, labelCounts = np.unique(y, return_counts=True)
    maxCount = np.max(labelCounts)
    labelInverseRatios = maxCount / labelCounts  
    # repeat for every label and concat
    newX = X[y == uniqueLabels[0], :, :, :].repeat(round(labelInverseRatios[0]), axis=0)
    newY = y[y == uniqueLabels[0]].repeat(round(labelInverseRatios[0]), axis=0)
    for label, labelInverseRatio in zip(uniqueLabels[1:], labelInverseRatios[1:]):
        cX = X[y == label,:,:,:].repeat(round(labelInverseRatio), axis=0)
        cY = y[y == label].repeat(round(labelInverseRatio), axis=0)
        newX = np.concatenate((newX, cX))
        newY = np.concatenate((newY, cY))
    np.random.seed(seed=42)
    rand_perm = np.random.permutation(newY.shape[0])
    newX = newX[rand_perm, :, :, :]
    newY = newY[rand_perm]
    return newX, newY

def shuffleTrainTest(train, test):
    np.random.seed(41)
    for i in range(train.shape[0]):
        for j in range(train.shape[1]):
            if train[i, j] != 0 or test[i, j] != 0 : #eviter calcul inutiles
                x = np.random.randint(1,3)
                if x == 1:
                    temp = train[i, j]
                    train[i, j] = test[i, j]
                    test[i, j] = temp
    return train, test

def deleteUselessClasses(data, classes_authorized):
    #data = data[data.any() in classes_authorized]
    #if not data
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] not in classes_authorized:
                data[i][j] = 0
            if data[i][j] == 2:
                data[i][j] = 1
            if data[i][j] == 3:
                data[i][j] = 2
            if data[i][j] == 5:
                data[i][j] = 3
            if data[i][j] == 6:
                data[i][j] = 4
            if data[i][j] == 10:
                data[i][j] = 5
            if data[i][j] == 11:
                data[i][j] = 6
            if data[i][j] == 12:
                data[i][j] = 7
            if data[i][j] == 14:
                data[i][j] = 8
            if data[i][j] == 15:
                data[i][j] = 9
    return data
