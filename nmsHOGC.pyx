import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

cdef float iOUThresh = 0.5

cpdef np.ndarray nms(np.ndarray boxes):
    cdef np.ndarray idxs = np.argsort(boxes[:, 0])
    chosenBoxes = []
    
    cdef int last
    cdef int i
    cdef int j
    while(len(idxs) > 0):
        last = len(idxs) - 1
        i = idxs[last]
        chosenBoxes.append(i)
        suppressedIndeces = [last]
        
        for j in range(last):
            if (calcIOU(boxes[i], boxes[idxs[j]]) > iOUThresh):
                suppressedIndeces.append(j)
                
        idxs = np.delete(idxs, suppressedIndeces)
    
    return(boxes[chosenBoxes])
    
cdef float calcIOU(np.ndarray boxA, np.ndarray boxB):
    cdef int y0A = boxA[1]
    cdef int y1A = boxA[2]
    cdef int x0A = boxA[3]
    cdef int x1A = boxA[4]

    cdef int y0B = boxB[1]
    cdef int y1B = boxB[2]
    cdef int x0B = boxB[3]
    cdef int x1B = boxB[4]

    
    cdef int xx0 = max(x0A, x0B)
    cdef int xx1 = min(x1A, x1B)
    
    cdef int yy0 = max(y0A, y0B)
    cdef int yy1 = min(y1A, y1B)
    
    cdef int xIntersect = max(0, xx1-xx0)
    cdef int yIntersect = max(0, yy1-yy0)
    
    cdef float intersect = float(xIntersect * yIntersect)
    
    cdef int boxAArea = (y1A - y0A)*(x1A - x0A)
    cdef int boxBArea = (y1B - y0B)*(x1B - x0B)
    
    cdef float iou = intersect / float(boxAArea + boxBArea - intersect)
    
    return iou
    
