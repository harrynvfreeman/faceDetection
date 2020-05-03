import numpy as np
from numpy import linalg as LA
import h5py
from scipy import ndimage
import math
import sys

#Image size should be 60x60
binSize = 9
imageSize = 36
cellSize = 6
numCell = 6 #imageSize/cellSize
blockSize = 3
pixelStride = 6
cellStride = 1 #pixelStride/cellSize
numBlocks = 16 #calculated
gradientMaskX = np.array([[-1, 0, 1]])
gradientMaskY = np.array([[-1, 0, 1]]).T
maxNormFactor = 0.2
epsilon = 0.001

def processHogToSave(mine):
    if (mine):
        trainDataset = h5py.File('dataset_train_LFW36Mine.h5', "r")
    else:
        trainDataset = h5py.File('dataset_train_LFW36.h5', "r")   
    #trainDataset = h5py.File('dataset_train_LFW36.h5', "r")
    #trainDataset = h5py.File('dataset_train_LFW36Mine.h5', "r")
    xTrain = np.array(trainDataset['dataset_train_image']).astype(np.float32)
    yTrain = np.array(trainDataset['dataset_train_label']).astype(np.float32)
    
    testDataset = h5py.File('dataset_test_LFW36.h5', "r")
    xTest = np.array(testDataset['dataset_test_image']).astype(np.float32)
    yTest = np.array(testDataset['dataset_test_label']).astype(np.float32)
    
    xTrainHog = []   
    for i in range(len(xTrain)):
        sys.stdout.flush()
        xTrainHog.append(calcHog(xTrain[i]))
    
    xTestHog = []
    for i in range(len(xTest)):
        sys.stdout.flush()
        xTestHog.append(calcHog(xTest[i]))
        
    if mine:
        h5f = h5py.File('hog_train_LFW36Mine.h5', 'w')
    else:
        h5f = h5py.File('hog_train_LFW36.h5', 'w')
        
    #h5f = h5py.File('hog_train_LFW36.h5', 'w')
    #h5f = h5py.File('hog_train_LFW36Mine.h5', 'w')
    h5f.create_dataset('hog_train_image', data=xTrainHog)
    h5f.create_dataset('hog_train_label', data=yTrain)
    h5f.close()

    h5f = h5py.File('hog_test_LFW36.h5', 'w')
    h5f.create_dataset('hog_test_image', data=xTestHog)
    h5f.create_dataset('hog_test_label', data=yTest)
    h5f.close()
    
    return xTrainHog, yTrain, xTestHog, yTest

def calcHog(x):

    gx = np.zeros(x.shape)
    gy = np.zeros(x.shape)
    
    gx[:,:,0] = ndimage.convolve(x[:,:,0], gradientMaskX)
    gy[:,:,0] = ndimage.convolve(x[:,:,0], gradientMaskY)
    gx[:,:,1] = ndimage.convolve(x[:,:,1], gradientMaskX)
    gy[:,:,1] = ndimage.convolve(x[:,:,1], gradientMaskY)
    gx[:,:,2] = ndimage.convolve(x[:,:,2], gradientMaskX)
    gy[:,:,2] = ndimage.convolve(x[:,:,2], gradientMaskY)
    
    vote = np.sqrt(np.square(gx) + np.square(gy))
    
    norm0 = LA.norm(vote[:,:,0])
    norm1 = LA.norm(vote[:,:,1])
    norm2 = LA.norm(vote[:,:,2])
    
    if norm0 <= norm1 and norm0 <= norm2:
        index = 0
    elif norm1 <= norm0 and norm1 <= norm2:
        index = 1
    else:
        index = 2
    
    vote = vote[:,:,index]
    gTheta = np.arctan2(gy[:,:,index], gx[:,:,index]) * 180 / math.pi
    gTheta[gTheta < 0] = gTheta[gTheta < 0] + 180
        
    histogram = castVotes(gTheta, vote)
    
    normalizedBlocks = normalize(histogram)
    
    return normalizedBlocks.ravel()

def castVotes(gTheta,vote):
    
    bz = 180/binSize;
    by = cellSize;
    bx = cellSize;
    
    cz = bz*np.arange(0.5, binSize + 0.5, 1)
    cy = by*np.arange(0.5, numCell + 0.5, 1) - 0.5
    cx = bx*np.arange(0.5, numCell + 0.5, 1) - 0.5
    
    gy = np.tile(np.arange(imageSize), (imageSize,1)).T
    gx = np.tile(np.arange(imageSize), (imageSize,1))
    
    histogram = np.zeros((numCell, numCell, binSize))
    
    bin0 = (np.mod(np.floor((gTheta-cz[0])/bz), binSize)).astype(np.int)
    bin1 = (np.mod(bin0 + 1, binSize)).astype(np.int)
    y0 = (np.mod(np.floor((gy-cy[0])/by), numCell)).astype(np.int)
    y1 = (np.mod(y0 + 1, numCell)).astype(np.int)
    x0 = (np.mod(np.floor((gx-cx[0])/bx), numCell)).astype(np.int)
    x1 = (np.mod(x0 + 1, numCell)).astype(np.int)
    
    binVote1 = np.mod(gTheta - cz[bin0], bz)/bz;
    binVote0 = 1 - binVote1;
    yVote1 = np.mod(gy - cy[y0], by)/by;
    yVote0 = 1 - yVote1;
    xVote1 = np.mod(gx - cx[x0], bx)/bx;
    xVote0 = 1 - xVote1;
    
    np.add.at(histogram, [y0.ravel(), x0.ravel(), bin0.ravel()], (vote*yVote0*xVote0*binVote0).ravel())
    np.add.at(histogram, [y0.ravel(), x0.ravel(), bin1.ravel()], (vote*yVote0*xVote0*binVote1).ravel())
    np.add.at(histogram, [y0.ravel(), x1.ravel(), bin0.ravel()], (vote*yVote0*xVote1*binVote0).ravel())
    np.add.at(histogram, [y0.ravel(), x1.ravel(), bin1.ravel()], (vote*yVote0*xVote1*binVote1).ravel())
    np.add.at(histogram, [y1.ravel(), x0.ravel(), bin0.ravel()], (vote*yVote1*xVote0*binVote0).ravel())
    np.add.at(histogram, [y1.ravel(), x0.ravel(), bin1.ravel()], (vote*yVote1*xVote0*binVote1).ravel())
    np.add.at(histogram, [y1.ravel(), x1.ravel(), bin0.ravel()], (vote*yVote1*xVote1*binVote0).ravel())
    np.add.at(histogram, [y1.ravel(), x1.ravel(), bin1.ravel()], (vote*yVote1*xVote1*binVote1).ravel())
            
    return histogram;

def normalize(histogram):
    blocks = np.zeros((numBlocks, binSize*blockSize*blockSize))
    
    blockIndex = 0
    rowStartIndex = 0
    rowEndIndex = blockSize
    
    
    while rowEndIndex <= numCell:
        colStartIndex = 0
        colEndIndex = blockSize
        
        while colEndIndex <= numCell:
            blocks[blockIndex, :] = histogram[rowStartIndex:rowEndIndex, colStartIndex:colEndIndex, :].ravel()
            blockIndex = blockIndex + 1
        
            colStartIndex = colStartIndex + cellStride
            colEndIndex = colEndIndex + cellStride
        rowStartIndex = rowStartIndex + cellStride
        rowEndIndex = rowEndIndex + cellStride
      
        
    normVals = LA.norm(blocks, axis=1, keepdims = True)
    blocks = blocks / np.sqrt(np.square(normVals) + np.square(epsilon))
    blocks[blocks > maxNormFactor] = maxNormFactor;
    normVals = LA.norm(blocks, axis=1, keepdims = True)
    blocks = blocks / np.sqrt(np.square(normVals) + np.square(epsilon))
    
    return blocks
    
    