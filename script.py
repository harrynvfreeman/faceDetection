from scipy import ndimage
import math
import numpy as np
from numpy import linalg as LA
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

x = np.zeros((imageSize,imageSize,3))
startVal = -1944
for i in range(3):
    for j in range(imageSize):
        for k in range(imageSize):
            x[j][k][i] = startVal*startVal
            startVal = startVal + 1
 
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

########
      


import time
start = time.time()
hogOutputs = hog.calcHog(imageSubsets[0:500])
end = time.time()
print(end - start)

image = Image.open('testPhoto.jpg')
npIm = np.array(image)
subNpIm = misc.imresize(image, [400, 200])
subIm = Image.fromarray(subNpIm)
boxes = detect.detect(subIm)
d = ImageDraw.Draw(subIm)
for (y0, y1, x0, x1) in boxes:
    d.rectangle(((x0,y0),(x1,y1)), outline = (0, 0, 255))
    
for j in range(last):
    if (calcIOU(boxes[i], boxes[idxs[j]]) > iOUThresh):
        w,y0,y1,x0,x1 = boxes[idxs[j]]
        d.rectangle(((x0,y0),(x1,y1)), outline = (255, 0, 0))
        suppressedIndeces.append(j)
w,y0,y1,x0,x1 = boxes[i]
d.rectangle(((x0,y0),(x1,y1)), outline = (0, 255, 0))


    scaledImage = misc.imresize(npImage, 1/scale).astype(np.float32)
    scaledRow = misc.imresize(npRow, 1/scale, mode = 'F', interp = 'nearest').astype(np.float32)
    scaledCol = misc.imresize(npCol, 1/scale, mode = 'F', interp = 'nearest').astype(np.float32)
        
    rowHog = ((scaledImage.shape[0] - hN)//windowStride + 1)
    colHog = ((scaledImage.shape[1] - wN)//windowStride + 1)
    imageSubsets = np.zeros((rowHog*colHog, hN, wN, 3))
    subFaces = np.zeros((rowHog*colHog, 5))
    count = 0
        
    iStart = 0
    iEnd = hN
        
    while iEnd <= scaledImage.shape[0]:
        jStart = 0
        jEnd = wN
        while jEnd <= scaledImage.shape[1]:
                
            imageSubsets[count] = scaledImage[iStart:iEnd, jStart:jEnd, :]
            subFaces[count] = [0, scaledRow[iStart][0], scaledRow[iEnd-1][0], scaledCol[0][jStart], scaledCol[0][jEnd-1]]
            count = count + 1
                
            #hogOutput = hog.calcHog(scaledImage[iStart:iEnd, jStart:jEnd, :]);                
                
            #detectionVal = np.dot(hogOutput, w) + b
            #if detectionVal > detectThresh:
                #faces.append([detectionVal, scaledRow[iStart][0], scaledRow[iEnd-1][0], scaledCol[0][jStart], scaledCol[0][jEnd-1]])
                
            jStart = jStart + windowStride
            jEnd = jEnd + windowStride
                
        iStart = iStart + windowStride
        iEnd = iEnd + windowStride
    
    hogOutputs = hog.calcHog(imageSubsets)
    detectionVals = np.dot(hogOutputs, w) + b
    iFaces = detectionVals > detectThresh
    subFaces[iFaces, 0] = detectionVals[iFaces]
    faces.extend(subFaces[iFaces])
        
    scale = scale*sR

    
    clf.cv_results_
{'mean_fit_time': array([107.29614417, 151.42707276,  24.44480356,  40.92775814,
        37.80778488,  19.41175723]), 'std_fit_time': array([1.55784703, 2.9290858 , 0.38553978, 0.55451641, 0.6574186 ,
       0.52712137]), 'mean_score_time': array([50.16297857, 69.93196686,  6.03735065, 19.25766158,  5.99336139,
        6.99471537]), 'std_score_time': array([0.8785934 , 0.1763632 , 0.16687514, 0.81311046, 0.05510831,
       0.20529229]), 'param_C': masked_array(data=[0.001, 0.001, 10, 10, 1000, 1000],
             mask=[False, False, False, False, False, False],
       fill_value='?',
            dtype=object), 'param_kernel': masked_array(data=['linear', 'rbf', 'linear', 'rbf', 'linear', 'rbf'],
             mask=[False, False, False, False, False, False],
       fill_value='?',
            dtype=object), 'params': [{'C': 0.001, 'kernel': 'linear'}, {'C': 0.001, 'kernel': 'rbf'}, {'C': 10, 'kernel': 'linear'}, {'C': 10, 'kernel': 'rbf'}, {'C': 1000, 'kernel': 'linear'}, {'C': 1000, 'kernel': 'rbf'}], 'split0_test_score': array([0.93927965, 0.53648878, 0.96303979, 0.96417122, 0.95229116,
       0.96549123]), 'split1_test_score': array([0.93983402, 0.53640136, 0.96454168, 0.96982271, 0.95247077,
       0.97170879]), 'split2_test_score': array([0.93623845, 0.53650255, 0.95868704, 0.95963026, 0.94661385,
       0.96359178]), 'mean_test_score': array([0.9384509 , 0.53646423, 0.96208978, 0.96454168, 0.95045895,
       0.96693072]), 'std_test_score': array([1.58050909e-03, 4.48092100e-05, 2.48267105e-03, 4.16909410e-03,
       2.71950040e-03, 3.46645415e-03]), 'rank_test_score': array([5, 6, 3, 2, 4, 1], dtype=int32), 'split0_train_score': array([0.93897953, 0.53645195, 0.98453268, 0.9679336 , 1.        ,
       0.98142035]), 'split1_train_score': array([0.9370049 , 0.53649566, 0.98453414, 0.96416447, 1.        ,
       0.98047906]), 'split2_train_score': array([0.94021688, 0.53644507, 0.98755304, 0.96954267, 1.        ,
       0.98227251]), 'mean_train_score': array([0.93873377, 0.53646423, 0.98553995, 0.96721358, 1.        ,
       0.98139064]), 'std_train_score': array([1.32274841e-03, 2.24039983e-05, 1.42346843e-03, 2.25389902e-03,
       0.00000000e+00, 7.32473585e-04])}