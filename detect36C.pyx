import numpy as np
cimport numpy as np
#import createHogFeatures36C as hog
from scipy import misc as misc
import nmsHOGC
import time
#from PIL import ImageDraw
#d = ImageDraw.Draw(image)
#for (y0, y1, x0, x1) in boxes:
#d.rectangle(((x0,y0),(x1,y1)), outline = (0, 255, 0))

#python3 setup.py build_ext --inplace
#python testing.py

from multiprocessing import Process, Queue

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cdef int hN = 36
cdef int wN = 36
cdef float sR = 1.3
cdef int windowStride = 6

cdef float detectThresh = 0.5

cdef extern from "hog.h":
    void calcHog(float * gTheta, float * subVote, int colSize, int rowStart, int colStart, float * normalizedBlocks);
    void prepHog(float * x, int rowSize, int colSize, float * gTheta, float * subVote);

cpdef np.ndarray detect(np.ndarray image):
    #this could be removed for speed
    w = np.load('wSave.npy').ravel()
    b = np.load('bSave.npy').ravel()
    
    npImage = np.array(image).astype(np.float32)
    npRow = np.tile(np.arange(npImage.shape[0]), (npImage.shape[1],1)).T
    npCol = np.tile(np.arange(npImage.shape[1]), (npImage.shape[0], 1))
    
    cdef int hI = npImage.shape[0]
    cdef int wI = npImage.shape[1]
    
    cdef int sS = 1
    cdef float sE = min(wI/wN, hI/hN)
    cdef float sN = np.floor(np.log(sE/sS)/np.log(sR) + 1)
    
    faces = []
        
    cdef float scale = sS
    
    cdef int scaleIndex
    cdef int iStart
    cdef int iEnd
    cdef int jStart
    cdef int jEnd
    
    cdef int scaleRow
    cdef int scaleCol
    cdef np.ndarray[DTYPE_t] hogOutput
    cdef float detectionVal
    
    cdef np.ndarray scaledImage #do I need to do np.ndarray[float, ndim=1, mode='c']??? Answer is YES
    cdef np.ndarray scaledRow
    cdef np.ndarray scaledCol
    
    cdef np.ndarray[DTYPE_t, ndim=3] newScaledImage
    cdef float * newScaledImagePointer
    
    cdef np.ndarray[DTYPE_t] gTheta #will there be issues converting np.float32 (which apprently is double) to my c code which expects float?
    cdef np.ndarray[DTYPE_t] subVote
    cdef float * gThetaPointer
    cdef float * subVotePointer
    
    q = Queue()
    cdef float numPartition = sN / 4

    '''
    cdef int core0 = np.ceil(numPartition).astype('int')
    cdef int core1 = np.ceil(numPartition - 0.25).astype('int')
    cdef int core2 = np.ceil(numPartition - 0.5).astype('int')
    cdef int core3 = int(sN) - core0 - core1 - core2

    p0 = Process(target = doScaleHOG, args=(q, npImage, npRow, npCol, scale, core0, sR, w, b))
    p1 = Process(target = doScaleHOG, args=(q, npImage, npRow, npCol, scale*np.power(sR, core0), core1, sR, w, b))
    p2 = Process(target = doScaleHOG, args=(q, npImage, npRow, npCol, scale*np.power(sR, core0+core1), core2, sR, w, b))
    p3 = Process(target = doScaleHOG, args=(q, npImage, npRow, npCol, scale*np.power(sR, core0+core1+core2), core3, sR, w, b))

    p0.start()
    p1.start()
    p2.start()
    p3.start()
    '''
    
    for scaleIndex in range(int(sN)):
        tempScale = scale
        tempW = w
        tempB = b
        tempImage = np.array(npImage)
        tempRow = np.array(npRow)
        tempCol = np.array(npCol)
        p = Process(target = doScaleHOG, args=(q, tempImage, tempRow, tempCol, tempScale, 1, sR, tempW, tempB))
        p.start()
        scale = scale*sR
    

    faces = np.array([[],[],[],[],[]]).T

    for scaleIndex in range(int(sN)):
        faces = np.concatenate((faces, q.get()))

    cdef np.ndarray newBoxes

    if faces.shape[0] > 0:
        newBoxes = nmsHOGC.nms(faces)

        #print('Has Faces: ' + str(np.array(newBoxes).shape[0]))	
    else:
        newBoxes = faces
        #print('No Faces')
    #end = time.time()
    #print('NmsTime:' + str(end-start))
    #return True
    return newBoxes
    	
    '''
    for scaleIndex in range(sN):
        start = time.time()
        newScaledImage = np.ascontiguousarray(np.transpose(misc.imresize(npImage, 1/scale).astype(DTYPE), (2, 0, 1)), dtype = DTYPE)
        scaledRow = misc.imresize(npRow, 1/scale, mode = 'F', interp = 'nearest').astype(np.float32)
        scaledCol = misc.imresize(npCol, 1/scale, mode = 'F', interp = 'nearest').astype(np.float32)
        newScaledImagePointer = <float *> newScaledImage.data
        
        #make gTheta and subVote
        gTheta = np.zeros((scaledRow.shape[0]*scaledCol.shape[1]), dtype=DTYPE) #is there an issue converting dtype to float
        subVote = np.zeros((scaledRow.shape[0]*scaledCol.shape[1]), dtype=DTYPE)
        gThetaPointer = <float *> gTheta.data
        subVotePointer = <float *> subVote.data
        
        scaleRow = scaledRow.shape[0] #both of these were scaledImage.shape[0/1]
        scaleCol = scaledCol.shape[1]
        
        prepHog(newScaledImagePointer, scaleRow, scaleCol, gThetaPointer, subVotePointer)
        end = time.time()
        #print('Hog Calc Time:' + str(end-start))
        iStart = 0
        iEnd = hN
        while iEnd < scaleRow:
            jStart = 0
            jEnd = wN
            while jEnd < scaleCol:
                hogOutput = np.zeros(1296, dtype = DTYPE)
                
                calcHog(gThetaPointer, subVotePointer, scaleCol, iStart, jStart, <float *> hogOutput.data);
                
                detectionVal = np.dot(hogOutput, w) + b
                if detectionVal > detectThresh:
                    faces.append([detectionVal, scaledRow[iStart][0], scaledRow[iEnd][0], scaledCol[0][jStart], scaledCol[0][jEnd]])
                
                jStart = jStart + windowStride
                jEnd = jEnd + windowStride
                
            iStart = iStart + windowStride
            iEnd = iEnd + windowStride
            
        scale = scale*sR
       
    #start = time.time()
    cdef np.ndarray newBoxes

    if len(faces) > 0:
        newBoxes = nmsHOGC.nms(np.array(faces))

        #print('Has Faces: ' + str(np.array(newBoxes).shape[0]))	
    else:
        newBoxes = np.array(faces)
        #print('No Faces')
    #end = time.time()
    #print('NmsTime:' + str(end-start))
    #return True
    return newBoxes
    '''


cpdef np.ndarray scaleHOG(np.ndarray npImage, np.ndarray npRow, np.ndarray npCol, float scale, int numIter, float sR, np.ndarray w, float b):
    cdef np.ndarray[DTYPE_t, ndim=3] newScaledImage
    cdef np.ndarray scaledRow
    cdef np.ndarray scaledCol
    cdef float * newScaledImagePointer

    cdef np.ndarray[DTYPE_t] gTheta
    cdef np.ndarray[DTYPE_t] subVote
    cdef float * gThetaPointer
    cdef float * subVotePointer

    cdef int scaleRow
    cdef int scaleCol

    cdef np.ndarray[DTYPE_t] hogOutput
    cdef float detectionVal

    cdef int iStart
    cdef int iEnd
    cdef int jStart
    cdef int jEnd

    cdef float subScale = scale

    faces = np.array([[],[],[],[],[]]).T
    for i in range(numIter):
        newScaledImage = np.ascontiguousarray(np.transpose(misc.imresize(npImage, 1/subScale).astype(DTYPE), (2, 0, 1)), dtype = DTYPE)
        scaledRow = misc.imresize(npRow, 1/scale, mode = 'F', interp = 'nearest').astype(np.float32)
        scaledCol = misc.imresize(npCol, 1/scale, mode = 'F', interp = 'nearest').astype(np.float32)
        newScaledImagePointer = <float *> newScaledImage.data

        gTheta = np.zeros((scaledRow.shape[0]*scaledCol.shape[1]), dtype=DTYPE)
        subVote = np.zeros((scaledRow.shape[0]*scaledCol.shape[1]), dtype=DTYPE)
        gThetaPointer = <float *> gTheta.data
        subVotePointer = <float *> subVote.data

        scaleRow = scaledRow.shape[0]
        scaleCol = scaledCol.shape[1]
        prepHog(newScaledImagePointer, scaleRow, scaleCol, gThetaPointer, subVotePointer)   

        iStart = 0
        iEnd = hN
	
        while iEnd < scaleRow:
            jStart = 0
            jEnd = wN
            while jEnd < scaleCol:
                hogOutput = np.zeros(1296, dtype = DTYPE)
                calcHog(gThetaPointer, subVotePointer, scaleCol, iStart, jStart, <float *> hogOutput.data);
                detectionVal = np.dot(hogOutput, w) + b

                if detectionVal > detectThresh:
                    faces = np.concatenate((faces, [[detectionVal, (scaledRow[iStart][0]), (scaledRow[iEnd][0]), (scaledCol[0][jStart]), (scaledCol[0][jEnd])]]))

                jStart = jStart + windowStride
                jEnd = jEnd + windowStride
            
            iStart = iStart + windowStride
            iEnd = iEnd + windowStride 

        subScale = subScale*sR

    return faces

def doScaleHOG(q, npImage, npRow, npCol, scale, numIter, sR, w, b):
    q.put(scaleHOG(npImage, npRow, npCol, scale, numIter, sR, w, b))
			
