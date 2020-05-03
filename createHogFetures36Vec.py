import numpy as np
from numpy import linalg as LA
from scipy import ndimage
import math

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

def calcHog(x):

    gx = np.zeros(x.shape)
    gy = np.zeros(x.shape)
    
    gx[:,:,:,0] = ndimage.convolve(x[:,:,:,0], np.expand_dims(gradientMaskX,axis=0))
    gy[:,:,:,0] = ndimage.convolve(x[:,:,:,0], np.expand_dims(gradientMaskY,axis=0))
    gx[:,:,:,1] = ndimage.convolve(x[:,:,:,1], np.expand_dims(gradientMaskX,axis=0))
    gy[:,:,:,1] = ndimage.convolve(x[:,:,:,1], np.expand_dims(gradientMaskY,axis=0))
    gx[:,:,:,2] = ndimage.convolve(x[:,:,:,2], np.expand_dims(gradientMaskX,axis=0))
    gy[:,:,:,2] = ndimage.convolve(x[:,:,:,2], np.expand_dims(gradientMaskY,axis=0))
    
    largVote = np.sqrt(np.square(gx) + np.square(gy))
    
    norm0 = LA.norm(largVote[:,:,:,0], axis = (1,2))
    norm1 = LA.norm(largVote[:,:,:,1], axis = (1,2))
    norm2 = LA.norm(largVote[:,:,:,2], axis = (1,2))
        
    index = np.zeros(largVote.shape[0])
    index[np.bitwise_and(norm1<=norm0, norm1<=norm2)] = 1
    index[np.bitwise_and(norm2<=norm0, norm2<=norm1)] = 2
    
    vote = np.zeros(largVote[:,:,:,0].shape)
    vote[index==0] = largVote[index==0,:,:,0]
    vote[index==1] = largVote[index==1,:,:,1]
    vote[index==2] = largVote[index==2,:,:,2]
    
    gTheta = np.zeros(vote.shape)
    gTheta[index==0] = np.arctan2(gy[index==0,:,:,0], gx[index==0,:,:,0]) * 180 / math.pi
    gTheta[index==1] = np.arctan2(gy[index==1,:,:,1], gx[index==1,:,:,1]) * 180 / math.pi
    gTheta[index==2] = np.arctan2(gy[index==2,:,:,2], gx[index==2,:,:,2]) * 180 / math.pi
    
    gTheta[gTheta < 0] = gTheta[gTheta < 0] + 180
        
    histogram = castVotes(gTheta, vote)
    
    normalizedBlocks = normalize(histogram)
    
    return normalizedBlocks.reshape(len(vote), -1)

def castVotes(gTheta,vote):
    
    bz = 180/binSize;
    by = cellSize;
    bx = cellSize;
    
    cz = bz*np.arange(0.5, binSize + 0.5, 1)
    cy = by*np.arange(0.5, numCell + 0.5, 1) - 0.5
    cx = bx*np.arange(0.5, numCell + 0.5, 1) - 0.5
    
    gx = np.tile(np.arange(imageSize), (len(vote),imageSize,1))
    gy = np.swapaxes(gx, 1, 2)
    #gy = np.tile(np.arange(imageSize), (imageSize,1)).T
    #gx = np.tile(np.arange(imageSize), (imageSize,1))
    rowArray = np.tile(np.arange(len(vote)), (imageSize*imageSize,1)).T
    
    histogram = np.zeros((len(vote),numCell, numCell, binSize))
    
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
    
    np.add.at(histogram, [rowArray.ravel(), y0.ravel(), x0.ravel(), bin1.ravel()], (vote*yVote0*xVote0*binVote1).ravel())
    np.add.at(histogram, [rowArray.ravel() ,y0.ravel(), x1.ravel(), bin0.ravel()], (vote*yVote0*xVote1*binVote0).ravel())
    np.add.at(histogram, [rowArray.ravel() ,y0.ravel(), x1.ravel(), bin1.ravel()], (vote*yVote0*xVote1*binVote1).ravel())
    np.add.at(histogram, [rowArray.ravel() ,y1.ravel(), x0.ravel(), bin0.ravel()], (vote*yVote1*xVote0*binVote0).ravel())
    np.add.at(histogram, [rowArray.ravel() ,y1.ravel(), x0.ravel(), bin1.ravel()], (vote*yVote1*xVote0*binVote1).ravel())
    np.add.at(histogram, [rowArray.ravel() ,y1.ravel(), x1.ravel(), bin0.ravel()], (vote*yVote1*xVote1*binVote0).ravel())
    np.add.at(histogram, [rowArray.ravel() ,y1.ravel(), x1.ravel(), bin1.ravel()], (vote*yVote1*xVote1*binVote1).ravel())
            
    return histogram;

def normalize(histogram):
    blocks = np.zeros((len(histogram), numBlocks, binSize*blockSize*blockSize))
    
    blockIndex = 0
    rowStartIndex = 0
    rowEndIndex = blockSize
    
    
    while rowEndIndex <= numCell:
        colStartIndex = 0
        colEndIndex = blockSize
        
        while colEndIndex <= numCell:
            blocks[:, blockIndex, :] = histogram[:, rowStartIndex:rowEndIndex, colStartIndex:colEndIndex, :].reshape(len(histogram), -1)
            blockIndex = blockIndex + 1
        
            colStartIndex = colStartIndex + cellStride
            colEndIndex = colEndIndex + cellStride
        rowStartIndex = rowStartIndex + cellStride
        rowEndIndex = rowEndIndex + cellStride
      
        
    normVals = LA.norm(blocks, axis=2, keepdims = True)
    blocks = blocks / np.sqrt(np.square(normVals) + np.square(epsilon))
    blocks[blocks > maxNormFactor] = maxNormFactor;
    normVals = LA.norm(blocks, axis=2, keepdims = True)
    blocks = blocks / np.sqrt(np.square(normVals) + np.square(epsilon))
    
    return blocks