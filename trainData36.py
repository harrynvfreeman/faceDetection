import numpy as np
import h5py
from sklearn import svm
from sklearn.metrics import accuracy_score
from joblib import dump, load
from sklearn.model_selection import GridSearchCV

def trainLFW():
    #trainDataset = h5py.File('hog_train_LFW36.h5', 'r')
    trainDataset = h5py.File('hog_train_LFW36Mine.h5', 'r')
    xTrain = np.array(trainDataset['hog_train_image'])
    yTrain = np.array(trainDataset['hog_train_label']).ravel()
    
    testDataset = h5py.File('hog_test_LFW36.h5', 'r')
    xTest = np.array(testDataset['hog_test_image'])
    yTest = np.array(testDataset['hog_test_label']).ravel()
    
    clf = svm.SVC(C=0.01, kernel='linear')
    
    clf.fit(xTrain, yTrain)
    
    train = clf.predict(xTrain)
    test = clf.predict(xTest)
    
    trainScore = accuracy_score(train, yTrain)
    testScore = accuracy_score(test, yTest)
    
    return clf, trainScore, testScore
    
    #dump(clf, 'faceClassifierLFW36.joblib')
    
def modelSelct():
    trainDataset = h5py.File('hog_train.h5', "r")
    xTrain = np.array(trainDataset['hog_train_image'])
    yTrain = np.array(trainDataset['hog_train_label'])
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10, 100, 1000, 10000]}
    svc = svm.SVC(gamma="scale")
    clf = GridSearchCV(svc, parameters, cv=5)
    clf.fit(xTrain, yTrain)
    return clf
    
