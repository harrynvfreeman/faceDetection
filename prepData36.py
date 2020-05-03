import numpy as np
import h5py
import os
from PIL import Image
import random
from scipy import misc
from joblib import dump, load
import createHogFeatures36 as hog

posPath = './LFW10/images'
negTrainPath = './INRIAPerson/train_64x128_H96/neg'
negNegativeMinePath = './INRIAPerson/test_64x128_H96/neg'

trainPercent = 70
testPercent = 30

rowSize = 36
colSize = 36

boundaryRow = 16
boundaryCol = 16
    
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
            npIm = npIm[52:198, 52:198, :]
            im = Image.fromarray(npIm)
            im.close()
            npIm = misc.imresize(npIm, [36, 36])
            setType = getSetType(trainPercent)
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
                rowStart = random.randint(0, shape[0]-36)
                rowEnd = rowStart + 36
                colStart = random.randint(0, shape[1]-36)
                colEnd = colStart + 36
                npImCrop = npIm[rowStart:rowEnd, colStart:colEnd, :]
                im = Image.fromarray(npImCrop)
                im.close()
                setType = getSetType(trainPercent)
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
    
    h5f = h5py.File('dataset_train_LFW36.h5', 'w')
    h5f.create_dataset('dataset_train_image', data=xTrain)
    h5f.create_dataset('dataset_train_label', data=yTrain)
    h5f.close()
    
    h5f = h5py.File('dataset_test_LFW36.h5', 'w')
    h5f.create_dataset('dataset_test_image', data=xTest)
    h5f.create_dataset('dataset_test_label', data=yTest)
    h5f.close()
    
    return xTrain, yTrain, xTest, yTest
    
    
    
def shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
       
            
def getSetType(trainPercent):
    randomVal = random.randrange(100)
    if randomVal < trainPercent:
        return 0
    else:
        return 1
    
    
def hardNegativeMine():
    trainDataset = h5py.File('dataset_train_LFW36.h5', "r")
    xTrain = np.array(trainDataset['dataset_train_image']).astype(np.float32)
    yTrain = np.array(trainDataset['dataset_train_label']).astype(np.float32)
    
    clf = load('faceClassifierLFW36.joblib')
    
    xTrainMine = []
    yTrainMine = []
    for filename in os.listdir(negNegativeMinePath):
        if (filename.endswith('.png')):
            im = Image.open(negNegativeMinePath + '/' + filename)
            npIm = np.array(im)
            im.close()
            shape = npIm.shape
            for i in range(10):
                rowStart = random.randint(0, shape[0]-36)
                rowEnd = rowStart + 36
                colStart = random.randint(0, shape[1]-36)
                colEnd = colStart + 36
                npImCrop = npIm[rowStart:rowEnd, colStart:colEnd, :]
                im = Image.fromarray(npImCrop)
                im.close()
                
                if (clf.predict(hog.calcHog(npImCrop).reshape(1,-1)) == 0):
                    xTrainMine.append(npImCrop)
                    yTrainMine.append([0])
    
    xTrainMine = np.array(xTrainMine)
    yTrainMine = np.array(yTrainMine)
    
    newXTrain = np.concatenate((xTrain, xTrainMine))
    newYTrain = np.concatenate((yTrain, yTrainMine))
    
    h5f = h5py.File('dataset_train_LFW36Mine.h5', 'w')
    h5f.create_dataset('dataset_train_image', data=newXTrain)
    h5f.create_dataset('dataset_train_label', data=newYTrain)
    h5f.close()
    
    
