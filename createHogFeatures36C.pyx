import numpy as np
cimport numpy as np
from numpy import linalg as LA
from scipy import ndimage
import math
import sys

DTYPE = np.float
ctypedef np.float_t DTYPE_t

#Image size should be 60x60
cdef int binSize = 9
cdef int imageSize = 36
cdef int cellSize = 6
cdef int numCell = 6 #imageSize/cellSize
cdef int blockSize = 3
cdef int pixelStride = 6
cdef int cellStride = 1 #pixelStride/cellSize
cdef int numBlocks = 16 #calculated
cdef np.ndarray gradientMaskX = np.array([[-1, 0, 1]])
cdef np.ndarray gradientMaskY = np.array([[-1, 0, 1]]).T
cdef float maxNormFactor = 0.2
cdef float epsilon = 0.001

cpdef np.ndarray calcHog(np.ndarray x):

    cdef np.ndarray gx = np.zeros((x.shape[0], x.shape[1], x.shape[2]))
    cdef np.ndarray gy = np.zeros((x.shape[0], x.shape[1], x.shape[2]))
    
    gx[:,:,0] = ndimage.convolve(x[:,:,0], gradientMaskX)
    gy[:,:,0] = ndimage.convolve(x[:,:,0], gradientMaskY)
    gx[:,:,1] = ndimage.convolve(x[:,:,1], gradientMaskX)
    gy[:,:,1] = ndimage.convolve(x[:,:,1], gradientMaskY)
    gx[:,:,2] = ndimage.convolve(x[:,:,2], gradientMaskX)
    gy[:,:,2] = ndimage.convolve(x[:,:,2], gradientMaskY)
    
    cdef np.ndarray vote = np.sqrt(np.square(gx) + np.square(gy))
    
    cdef float norm0 = LA.norm(vote[:,:,0])
    cdef float norm1 = LA.norm(vote[:,:,1])
    cdef float norm2 = LA.norm(vote[:,:,2])
    
    cdef int index
    if norm0 <= norm1 and norm0 <= norm2:
        index = 0
    elif norm1 <= norm0 and norm1 <= norm2:
        index = 1
    else:
        index = 2
    
    vote = vote[:,:,index]
    cdef np.ndarray gTheta = np.arctan2(gy[:,:,index], gx[:,:,index]) * 180 / math.pi
    gTheta[gTheta < 0] = gTheta[gTheta < 0] + 180
        
    cdef np.ndarray histogram = castVotes(gTheta, vote)
    
    cdef np.ndarray normalizedBlocks = normalize(histogram)
    
    return normalizedBlocks.ravel()

cdef np.ndarray castVotes(np.ndarray gTheta, np.ndarray vote):
    
    cdef float bz = 180/binSize;
    cdef int by = cellSize;
    cdef int bx = cellSize;
    
    cdef np.ndarray cz = bz*np.arange(0.5, binSize + 0.5, 1)
    cdef np.ndarray cy = by*np.arange(0.5, numCell + 0.5, 1) - 0.5
    cdef np.ndarray cx = bx*np.arange(0.5, numCell + 0.5, 1) - 0.5
    
    cdef np.ndarray gy = np.tile(np.arange(imageSize), (imageSize,1)).T
    cdef np.ndarray gx = np.tile(np.arange(imageSize), (imageSize,1))
    
    cdef np.ndarray histogram = np.zeros((numCell, numCell, binSize))
    
    cdef np.ndarray bin0 = (np.mod(np.floor((gTheta-cz[0])/bz), binSize)).astype(np.int)
    cdef np.ndarray bin1 = (np.mod(bin0 + 1, binSize)).astype(np.int)
    cdef np.ndarray y0 = (np.mod(np.floor((gy-cy[0])/by), numCell)).astype(np.int)
    cdef np.ndarray y1 = (np.mod(y0 + 1, numCell)).astype(np.int)
    cdef np.ndarray x0 = (np.mod(np.floor((gx-cx[0])/bx), numCell)).astype(np.int)
    cdef np.ndarray x1 = (np.mod(x0 + 1, numCell)).astype(np.int)
    
    cdef np.ndarray binVote1 = np.mod(gTheta - cz[bin0], bz)/bz;
    cdef np.ndarray binVote0 = 1 - binVote1;
    cdef np.ndarray yVote1 = np.mod(gy - cy[y0], by)/by;
    cdef np.ndarray yVote0 = 1 - yVote1;
    cdef np.ndarray xVote1 = np.mod(gx - cx[x0], bx)/bx;
    cdef np.ndarray xVote0 = 1 - xVote1;
    
    np.add.at(histogram, [y0.ravel(), x0.ravel(), bin0.ravel()], (vote*yVote0*xVote0*binVote0).ravel())
    np.add.at(histogram, [y0.ravel(), x0.ravel(), bin1.ravel()], (vote*yVote0*xVote0*binVote1).ravel())
    np.add.at(histogram, [y0.ravel(), x1.ravel(), bin0.ravel()], (vote*yVote0*xVote1*binVote0).ravel())
    np.add.at(histogram, [y0.ravel(), x1.ravel(), bin1.ravel()], (vote*yVote0*xVote1*binVote1).ravel())
    np.add.at(histogram, [y1.ravel(), x0.ravel(), bin0.ravel()], (vote*yVote1*xVote0*binVote0).ravel())
    np.add.at(histogram, [y1.ravel(), x0.ravel(), bin1.ravel()], (vote*yVote1*xVote0*binVote1).ravel())
    np.add.at(histogram, [y1.ravel(), x1.ravel(), bin0.ravel()], (vote*yVote1*xVote1*binVote0).ravel())
    np.add.at(histogram, [y1.ravel(), x1.ravel(), bin1.ravel()], (vote*yVote1*xVote1*binVote1).ravel())
            
    return histogram;

cdef np.ndarray normalize(np.ndarray histogram):
    cdef np.ndarray blocks = np.zeros((numBlocks, binSize*blockSize*blockSize))
    
    cdef int blockIndex = 0
    cdef int rowStartIndex = 0
    cdef int rowEndIndex = blockSize
    
    cdef int colStartIndex
    cdef int colEndIndex
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
      
        
    cdef np.ndarray normVals = LA.norm(blocks, axis=1, keepdims = True)
    blocks = blocks / np.sqrt(np.square(normVals) + np.square(epsilon))
    blocks[blocks > maxNormFactor] = maxNormFactor;
    normVals = LA.norm(blocks, axis=1, keepdims = True)
    blocks = blocks / np.sqrt(np.square(normVals) + np.square(epsilon))
    
    return blocks
    
    
