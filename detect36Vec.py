import numpy as np
from joblib import dump, load
from sklearn import svm
import createHogFetures36Vec as hog
import sys
from scipy import misc as misc
#import nms
#import nmsFast
import nmsHOG
from PIL import ImageDraw
import time
#from PIL import ImageDraw
#d = ImageDraw.Draw(image)
#for (y0, y1, x0, x1) in boxes:
#d.rectangle(((x0,y0),(x1,y1)), outline = (0, 255, 0))

###NO DETECT2

hN = 36
wN = 36
sR = 1.2
windowStride = 12
hogSize = 1296

#.9 seems to work better
detectThresh = .7

def detect(image):
    
    clf = load('faceClassifierLFW36.joblib')
    
    w = clf.coef_[0];
    b = clf.intercept_[0]
    
    npImage = np.array(image).astype(np.float32)
    npRow = np.tile(np.arange(npImage.shape[0]), (npImage.shape[1],1)).T
    npCol = np.tile(np.arange(npImage.shape[1]), (npImage.shape[0], 1))
    
    hI = npImage.shape[0]
    wI = npImage.shape[1]
    
    sS = 1
    sE = min(wI/wN, hI/hN)
    sN = int(np.floor(np.log(sE/sS)/np.log(sR) + 1))
    
    faces = []
        
    scale = sS
    
    timeCount = 0
    start = time.time()
    for scaleIndex in range(sN):
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
                timeCount = timeCount + 1
                
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
    
        
    end = time.time()
    print('Hog Calc Time:' + str(end-start) + ', with count=' + str(timeCount))
    #return nms.non_max_suppression_slow(np.array(faces), 0.3)
    #return nmsFast.non_max_suppression_fast(np.array(faces), 0.3)
    start = time.time()
    if len(faces) > 0:
        newBoxes = nmsHOG.nms(np.array(faces))
    else:
        newBoxes = []
    end = time.time()
    print('NmsTime:' + str(end-start))
    d = ImageDraw.Draw(image)
    for (w, y0, y1, x0, x1) in newBoxes:
        d.rectangle(((x0,y0),(x1,y1)), outline = (0, 0, 255))
    image.show()
    
    return newBoxes
    #return np.array(faces)
        
        
    
    