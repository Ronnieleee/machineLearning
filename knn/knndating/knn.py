#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: ronnie

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import operator

def createDataSet():
    """ data load module

    for basic test
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(inX, dataSet, labels, k):
    """Classification

    inX: Unclassified vector
    dataSet: learning samples set
    labels: label vector for the dataSet
    k: counts of the neighbor
    """
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    #calculate the sum of the row
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),
            key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def testfile2matrix(filename):
    """ Read test data from the file and store it into the ndarray

    return the dataSet and the labelVector
    """
    fr = open(filename, 'r')
    arrayOLines = fr.readlines()

    # File lines
    numberOfLines = len(arrayOLines)
    # Numpy Matrix
    returnMat = np.zeros((numberOfLines, 3))
    index = 0

    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        index += 1

    return returnMat

def file2matrix(filename):
    """ Read learning data from the file and store it into the ndarray

    return the dataSet and the labelVector
    """
    fr = open(filename, 'r')
    arrayOLines = fr.readlines()

    # File lines
    numberOfLines = len(arrayOLines)
    # Numpy Matrix
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0

    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    return returnMat, classLabelVector

def autoNorm(dataSet):
    """Normalized the dataSet
    @dataSet: .
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet/np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

# plot the scatterplot
def datingShowPL():
    """Draw the scatterplot of P (as x) & L (as y)

    P: percentages of time spent playing video games
    L: liters of ice cream consumed per year
    """
    datingDataMat, datingLabel = file2matrix('datingTestSet.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
    ax.set_xlabel('percentages of time spent playing video games')
    ax.set_ylabel('liters of ice cream consumed per year')
    plt.show()

def datingShowFP():
    """Draw the scatterplot of F (as x) & P (as y)

    F: frequent flier miles earned per year
    P: percentages of time spent playing video games
    """
    datingDataMat, datingLabel = file2matrix('datingTestSet.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,0], datingDataMat[:,1])
    ax.set_xlabel('frequent flier miles earned per year')
    ax.set_ylabel('percentages of time spent playing video games')
    plt.show()

def datingShowFL():
    """Draw the scatterplot of F (as x) & P (as y)

    F: frequent flier miles earned per year
    L: liters of ice cream consumed per year
    """
    datingDataMat, datingLabel = file2matrix('datingTestSet.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,0], datingDataMat[:,2])
    ax.set_xlabel('frequent flier miles earned per year')
    ax.set_ylabel('liters of ice cream consumed per year')
    plt.show()

# learn from the 90% data then classify the remaining 10%
def datingClassTest(k):
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat,ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    print("With k = %d" % int(k))
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i,:], normMat[numTestVecs:m,:],\
                datingLabels[numTestVecs:m],int(k))
        if (classifierResult != datingLabels[i]):
        	errorCount += 1.0
        	print("the classifier came back with: %d, the real answer is: %d"\
        		% (classifierResult, datingLabels[i]))

    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

# Prediction
def classifyPerson(k):
    resultList = ['didntLike', 'smallDoses', 'largeDoses']
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    testdatingMat = testfile2matrix('testData.txt')
    resultfile = open("result.txt", 'w')
    for i in range(testdatingMat.shape[0]):
        result = classify((testdatingMat[i,:]-\
            minVals)/ranges, normMat, datingLabels,k)
        #print("You will probably like this person: ",resultList[result - 1])
        resultfile.write(resultList[result - 1] + '\n')
    resultfile.close()

def main():

# run python knn.py >> k-testresult k best be 4
#    for k in range(1, 901):
#        datingClassTest(k)
    classifyPerson(4)

if __name__ == '__main__':
    main()
