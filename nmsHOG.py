import numpy as np

iOUThresh = 0.5
def nms(boxes):
    idxs = np.argsort(boxes[:, 0])
    chosenBoxes = []
    
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
    
def calcIOU(boxA, boxB):
    wA, y0A, y1A, x0A, x1A = boxA
    wB, y0B, y1B, x0B, x1B = boxB
    
    xx0 = max(x0A, x0B)
    xx1 = min(x1A, x1B)
    
    yy0 = max(y0A, y0B)
    yy1 = min(y1A, y1B)
    
    xIntersect = max(0, xx1-xx0)
    yIntersect = max(0, yy1-yy0)
    
    intersect = float(xIntersect * yIntersect)
    
    boxAArea = (y1A - y0A)*(x1A - x0A)
    boxBArea = (y1B - y0B)*(x1B - x0B)
    
    iou = intersect / float(boxAArea + boxBArea - intersect)
    
    return iou
    