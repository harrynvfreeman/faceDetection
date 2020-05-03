import h5py
import os
from PIL import Image
import random
from scipy import misc
import itertools
import numpy as np

mainPath = './lfw'
federerPath = './lfw/Roger_Federer'
trainPath = './triplet/train'
testPath = './triplet/test'

tripletPath146 = './tripletPath146'

trainPercent = 70
testPercent = 30

#Need to generate A, P, N
#Find all files with two people

def getPaths():
    doubleList = []
    singleList = []
    for dir in os.listdir(mainPath):
        newPath = os.path.join(mainPath, dir)
        if os.path.isdir(newPath):
            files = [name for name in os.listdir(newPath) if name.endswith('.jpg')]
            if len(files) > 1:
                doubleList.append(newPath)
            elif len(files) == 1:
                file = os.path.join(newPath, files[0])
                singleList.append(file)
    
    return doubleList, singleList

def minSizePrep():
    for dir in os.listdir(mainPath):
        newPath = os.path.join(mainPath, dir)
        if os.path.isdir(newPath):
            for name in os.listdir(newPath):
                if name.endswith('.jpg'):
                    imagePath = os.path.join(newPath, name)
                    im = Image.open(imagePath)
                    npIm = np.array(im)
                    npIm = npIm[52:198, 52:198]
                    im = Image.fromarray(npIm)
                    
                    savePath = os.path.join(tripletPath146, name)
                    
                    im.save(savePath)
                    
                    im.close
                    

#get every combination of 2, and assign a random non anchor
def createTriplets():
    doubleList, singleList = getPaths()
    
    #random shuffle singleList
    p = np.random.permutation(len(singleList))
    trainCount = 0
    testCount = 0
    for dir in doubleList:
        files = [name for name in os.listdir(dir) if name.endswith('.jpg')]
        combinations = list(itertools.combinations(range(len(files)), 2))
        for comb in combinations:
            
            anchorIndex = np.random.randint(2)
            posIndex = 1 - anchorIndex
            negIndex = np.random.randint(len(singleList))
            
            anchorPath = os.path.join(dir, files[comb[anchorIndex]])
            posPath = os.path.join(dir, files[comb[posIndex]])
            negPath = singleList[negIndex]
            
            anchorIm = Image.open(anchorPath)
            npAnchorIm = np.array(anchorIm)
            npAnchorIm = misc.imresize(npAnchorIm[52:198, 52:198, :], [36, 36])
            anchorIm = Image.fromarray(npAnchorIm)
            
            posIm = Image.open(posPath)
            npPosIm = np.array(posIm)
            npPosIm = misc.imresize(npPosIm[52:198, 52:198, :], [36, 36])
            posIm = Image.fromarray(npPosIm)
            
            negIm = Image.open(negPath)
            npNegIm = np.array(negIm)
            npNegIm = misc.imresize(npNegIm[52:198, 52:198, :], [36, 36])
            negIm = Image.fromarray(npNegIm)
            
            setType = getSetType(trainPercent)
            if setType == 0:
                savePath = os.path.join(trainPath, str(trainCount))
                trainCount = trainCount + 1
            else:
                savePath = os.path.join(testPath, str(testCount))
                testCount = testCount + 1
            
            os.makedirs(savePath)
            anchorIm.save(os.path.join(savePath, 'anchor.jpg'))
            posIm.save(os.path.join(savePath, 'pos.jpg'))
            negIm.save(os.path.join(savePath, 'neg.jpg'))
            
            
def getSetType(trainPercent):
    randomVal = random.randrange(100)
    if randomVal < trainPercent:
        return 0
    else:
        return 1
    
    
def createH5PyFiles():
    ATrain = []
    PTrain = []
    NTrain = []
    
    ATest = []
    PTest = []
    NTest = []
    
    for dir in os.listdir(trainPath):
        subTPath = os.path.join(trainPath, dir)
        if os.path.isdir(subTPath):
            im = Image.open(os.path.join(subTPath, 'anchor.jpg'))
            ATrain.append(np.array(im))
            im.close()
            
            im = Image.open(os.path.join(subTPath, 'pos.jpg'))
            PTrain.append(np.array(im))
            im.close()
            
            im = Image.open(os.path.join(subTPath, 'neg.jpg'))
            NTrain.append(np.array(im))
            im.close()
    
    for dir in os.listdir(testPath):
        subTPath = os.path.join(testPath, dir)
        if os.path.isdir(subTPath):
            im = Image.open(os.path.join(subTPath, 'anchor.jpg'))
            ATest.append(np.array(im))
            im.close()
            
            im = Image.open(os.path.join(subTPath, 'pos.jpg'))
            PTest.append(np.array(im))
            im.close()
            
            im = Image.open(os.path.join(subTPath, 'neg.jpg'))
            NTest.append(np.array(im))
            im.close()
            
    ATrain = np.array(ATrain)
    PTrain = np.array(PTrain)
    NTrain = np.array(NTrain)
    ATrain, PTrain, NTrain = shuffle(ATrain, PTrain, NTrain)
    
    ATest = np.array(ATest)
    PTest = np.array(PTest)
    NTest = np.array(NTest)
    ATest, PTest, NTest = shuffle(ATest, PTest, NTest)
    
    #h5f = h5py.File('triplet_train.h5', 'w')
    #h5f.create_dataset('anchor', data=ATrain)
    #h5f.create_dataset('pos', data=PTrain)
    #h5f.create_dataset('neg', data=NTrain)
    #h5f.close()
    
    #h5f = h5py.File('triplet_test.h5', 'w')
    #h5f.create_dataset('anchor', data=ATest)
    #h5f.create_dataset('pos', data=PTest)
    #h5f.create_dataset('neg', data=NTest)
    #h5f.close()
    
    return ATrain, PTrain, NTrain, ATest, PTest, NTest

def minSizeCreateH5pyFiles():
    train = []
    test = []
    
    for dir in os.listdir(tripletPath146):
        if dir.endswith('.jpg'):
            imagePath = os.path.join(tripletPath146, dir)
            im = Image.open(imagePath)
            
            yA = np.random.randint(0, 36)
            yB = np.random.randint(0, 36)
            xA = np.random.randint(0, 36)
            xB = np.random.randint(0, 36)
            
            
            npIm = np.array(im)
            npImA = npIm[yA:yA + 110, xA: xA + 110]
            npImB = npIm[yB:yB + 110, xB: xB + 110]
            
            setType = getSetType(trainPercent)
            
            if setType == 0:
                train.append(npImA)
                train.append(npImB)
            else:
                test.append(npImA)
                test.append(npImB)
            
    train = np.array(train)
    test = np.array(test)
            
    h5f = h5py.File('triplet_train110.h5', 'w')
    h5f.create_dataset('train', data=train)
    h5f.close()
    
    h5f = h5py.File('triplet_test110.h5', 'w')
    h5f.create_dataset('test', data=test)
    h5f.close()
    

def shuffle(a, b, c):
    assert len(a) == len(b)
    assert len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]


def rogerPrep():
    rogerPhotos = []
    for filename in os.listdir(federerPath):
        if filename.endswith('.jpg'):
            rogerPhoto = np.array(Image.open(os.path.join(federerPath, filename)))
            npIm = np.array(misc.imresize(rogerPhoto[52:198, 52:198, :], [36, 36]))
            rogerPhotos.append(npIm)
    
    rogerPhotos = np.array(rogerPhotos)
    h5f = h5py.File('roger_train.h5', 'w')
    h5f.create_dataset('roger_photos', data=rogerPhotos)
    
    
            
        
        