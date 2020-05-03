import numpy as np
import h5py
import os
from PIL import Image
import random
from scipy import misc

posPath = './LFW10/images'
savePath = './LFW150/'
#posImagePath = './LFW10'
#posDescriptorPath = './cwLabel0.txt'
negTrainPath = './INRIAPerson/train_64x128_H96/neg'
negTestPath = './INRIAPerson/test_64x128_H96/neg'

trainPercent = 70
cvPercent = 0
testPercent = 30

rowSize = 36
colSize = 36

boundaryRow = 16
boundaryCol = 16

def readDesciptors(path):
    with open(path) as f:
        myFiles = list(f)
        
    mySplitFiles = []
    for file in myFiles:
        splitFile = file.split()
        for i in range(1, len(splitFile)):
            splitFile[i] = int(float(splitFile[i]))
            
        mySplitFiles.append(splitFile)
    
    return mySplitFiles

def createH5pyForImagesCalTech():
    
    mySplitFiles = readDesciptors(posDescriptorPath)
    
    xTrain = []
    yTrain = []
    xCV = []
    yCV = []
    xTest = []
    yTest = []
    
    #First collect positive images
    for i in range(len(mySplitFiles)):
        im = Image.open(posImagePath + '/' + mySplitFiles[i][0]).convert('L')
        npIm = np.array(im)
        im.close()
        lEyeX =  mySplitFiles[i][1]
        lEyeY =  mySplitFiles[i][2]
        rEyeX =  mySplitFiles[i][3]
        rEyeY =  mySplitFiles[i][4]
        noseX =  mySplitFiles[i][5]
        noseY =  mySplitFiles[i][6]
        mouthX =  mySplitFiles[i][7]
        mouthY =  mySplitFiles[i][8]
        
        x0 = min(lEyeX, rEyeX, noseX, mouthX)
        x1 = max(lEyeX, rEyeX, noseX, mouthX)
        y0 = min(lEyeY, rEyeY, noseY, mouthY)
        y1 = max(lEyeY, rEyeY, noseY, mouthY)        
    
        rowOffset = min(y0, boundaryRow, npIm.shape[0]-y1-1)
        colOffset = min(x0, boundaryCol, npIm.shape[1]-x1-1)
        npIm = npIm[y0-rowOffset:y1+rowOffset+1,x0-colOffset: x1+colOffset+1]
        npIm = misc.imresize(npIm, [rowSize, colSize])
        setType = getSetType(trainPercent, cvPercent)
        if setType == 0:
            xTrain.append(npIm)
            yTrain.append([1])
        elif setType == 1:
            xCV.append(npIm)
            yCV.append([1])
        else:
            xTest.append(npIm)
            yTest.append([1])
            
    
    #Now collect negative images
    for filename in os.listdir(negTrainPath):
        if (filename.endswith('.png')):
            im = Image.open(negTrainPath + '/' + filename).convert('L')
            npIm = np.array(im)
            im.close()
            shape = npIm.shape
            for i in range(10):
                rowStart = random.randint(0, shape[0]-rowSize)
                rowEnd = rowStart + rowSize
                colStart = random.randint(0, shape[1]-colSize)
                colEnd = colStart + colSize
                npImCrop = npIm[rowStart:rowEnd, colStart:colEnd]
                setType = getSetType(trainPercent, cvPercent)
                if setType == 0:
                    xTrain.append(npImCrop)
                    yTrain.append([0])
                elif setType == 1:
                    xCV.append(npImCrop)
                    yCV.append([0])
                else:
                    xTest.append(npImCrop)
                    yTest.append([0])
    
    for filename in os.listdir(negTestPath):
        if (filename.endswith('.png')):
            im = Image.open(negTestPath + '/' + filename).convert('L')
            npIm = np.array(im)
            im.close()
            shape = npIm.shape
            for i in range(10):
                rowStart = random.randint(0, shape[0]-rowSize)
                rowEnd = rowStart + rowSize
                colStart = random.randint(0, shape[1]-colSize)
                colEnd = colStart + colSize
                npImCrop = npIm[rowStart:rowEnd, colStart:colEnd]
                setType = getSetType(trainPercent, cvPercent)
                if setType == 0:
                    xTrain.append(npImCrop)
                    yTrain.append([0])
                elif setType == 1:
                    xCV.append(npImCrop)
                    yCV.append([0])
                else:
                    xTest.append(npImCrop)
                    yTest.append([0])
                
    
    xTrain = np.array(xTrain)
    yTrain = np.array(yTrain)
    xTrain, yTrain = shuffle(xTrain, yTrain)
    xCV = np.array(xCV)
    yCV = np.array(yCV)
    xTest = np.array(xTest)
    yTest = np.array(yTest)
    
    #return xTrain, yTrain, xCV, yCV, xTest, yTest
    
    h5f = h5py.File('dataset_train.h5', 'w')
    h5f.create_dataset('dataset_train_image', data=xTrain)
    h5f.create_dataset('dataset_train_label', data=yTrain)
    h5f.close()
    
    h5f = h5py.File('dataset_cv.h5', 'w')
    h5f.create_dataset('dataset_cv_image', data=xCV)
    h5f.create_dataset('dataset_cv_label', data=yCV)
    h5f.close()
    
    h5f = h5py.File('dataset_test.h5', 'w')
    h5f.create_dataset('dataset_test_image', data=xTest)
    h5f.create_dataset('dataset_test_label', data=yTest)
    h5f.close()
    

def newToRenameCalTech():
    xTrain = []
    yTrain = []
    xCV = []
    yCV = []
    xTest = []
    yTest = []
    for filename in os.listdir(posImagePath):
        if (filename.endswith('.jpg')):
            im = Image.open(posImagePath + '/' + filename).convert('L')
            npIm = np.array(im)
            im.close()
            setType = getSetType(trainPercent, cvPercent)
            if setType == 0:
                xTrain.append(npIm)
                yTrain.append([1])
            elif setType == 1:
                xCV.append(npIm)
                yCV.append([1])
            else:
                xTest.append(npIm)
                yTest.append([1])
                
    #Now collect negative images
    for filename in os.listdir(negTrainPath):
        if (filename.endswith('.png')):
            im = Image.open(negTrainPath + '/' + filename).convert('L')
            npIm = np.array(im)
            im.close()
            shape = npIm.shape
            for i in range(60):
                rowStart = random.randint(0, shape[0]-64)
                rowEnd = rowStart + 64
                colStart = random.randint(0, shape[1]-64)
                colEnd = colStart + 64
                npImCrop = npIm[rowStart:rowEnd, colStart:colEnd]
                setType = getSetType(trainPercent, cvPercent)
                if setType == 0:
                    xTrain.append(npImCrop)
                    yTrain.append([0])
                elif setType == 1:
                    xCV.append(npImCrop)
                    yCV.append([0])
                else:
                    xTest.append(npImCrop)
                    yTest.append([0])
    
    for filename in os.listdir(negTestPath):
        if (filename.endswith('.png')):
            im = Image.open(negTestPath + '/' + filename).convert('L')
            npIm = np.array(im)
            im.close()
            shape = npIm.shape
            for i in range(60):
                rowStart = random.randint(0, shape[0]-64)
                rowEnd = rowStart + 64
                colStart = random.randint(0, shape[1]-64)
                colEnd = colStart + 64
                npImCrop = npIm[rowStart:rowEnd, colStart:colEnd]
                setType = getSetType(trainPercent, cvPercent)
                if setType == 0:
                    xTrain.append(npImCrop)
                    yTrain.append([0])
                elif setType == 1:
                    xCV.append(npImCrop)
                    yCV.append([0])
                else:
                    xTest.append(npImCrop)
                    yTest.append([0])
                
    
    xTrain = np.array(xTrain)
    yTrain = np.array(yTrain)
    xTrain, yTrain = shuffle(xTrain, yTrain)
    xCV = np.array(xCV)
    yCV = np.array(yCV)
    xTest = np.array(xTest)
    yTest = np.array(yTest)
    
    #return xTrain, yTrain, xCV, yCV, xTest, yTest
    
    h5f = h5py.File('dataset_train.h5', 'w')
    h5f.create_dataset('dataset_train_image', data=xTrain)
    h5f.create_dataset('dataset_train_label', data=yTrain)
    h5f.close()
    
    h5f = h5py.File('dataset_cv.h5', 'w')
    h5f.create_dataset('dataset_cv_image', data=xCV)
    h5f.create_dataset('dataset_cv_label', data=yCV)
    h5f.close()
    
    h5f = h5py.File('dataset_test.h5', 'w')
    h5f.create_dataset('dataset_test_image', data=xTest)
    h5f.create_dataset('dataset_test_label', data=yTest)
    h5f.close()

    
def createH5pyForImagesLFW():
    xTrain = []
    yTrain = []
    xTest = []
    yTest = []
    
    for filename in os.listdir(posPath):
        if (filename.endswith('.jpg')):
            im = Image.open(posPath + '/' + filename)
            npIm = np.array(im)
            im.close()
            npIm = npIm[50:200, 50:200, :]
            im = Image.fromarray(npIm)
            #im.save(savePath + '/pos' + filename)
            im.close()
            npIm = misc.imresize(npIm, [60, 60])
            setType = getSetType(trainPercent, cvPercent)
            if setType == 0:
                xTrain.append(npIm)
                yTrain.append([1])
            else:
                xTest.append(npIm)
                yTest.append([1])
                
    for filename in os.listdir(negTrainPath):
        if (filename.endswith('.png')):
            im = Image.open(negTrainPath + '/' + filename)
            npIm = np.array(im)
            im.close()
            shape = npIm.shape
            for i in range(10):
                rowStart = random.randint(0, shape[0]-60)
                rowEnd = rowStart + 60
                colStart = random.randint(0, shape[1]-60)
                colEnd = colStart + 60
                npImCrop = npIm[rowStart:rowEnd, colStart:colEnd, :]
                im = Image.fromarray(npImCrop)
                #im.save(savePath + '/neg' + filename)
                im.close()
                setType = getSetType(trainPercent, cvPercent)
                if setType == 0:
                    xTrain.append(npImCrop)
                    yTrain.append([0])
                else:
                    xTest.append(npImCrop)
                    yTest.append([0])
                    
    xTrain = np.array(xTrain)
    yTrain = np.array(yTrain)
    xTrain, yTrain = shuffle(xTrain, yTrain)
    xTest = np.array(xTest)
    yTest = np.array(yTest)
    
    h5f = h5py.File('dataset_train_LFW3.h5', 'w')
    h5f.create_dataset('dataset_train_image', data=xTrain)
    h5f.create_dataset('dataset_train_label', data=yTrain)
    h5f.close()
    
    h5f = h5py.File('dataset_test_LFW3.h5', 'w')
    h5f.create_dataset('dataset_test_image', data=xTest)
    h5f.create_dataset('dataset_test_label', data=yTest)
    h5f.close()
    
    return xTrain, yTrain, xTest, yTest
    
    
    
def shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
       
            
def getSetType(trainPercent, cvPercent):
    randomVal = random.randrange(100)
    if randomVal < trainPercent:
        return 0
    elif  randomVal < trainPercent + cvPercent:
        return 1
    else:
        return 2