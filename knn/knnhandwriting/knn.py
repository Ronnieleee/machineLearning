#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: ronnie
#
# Copyright 2015 Oracle Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import operator
from os import listdir
from time import *

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

def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect;

def handwritingclasstest(k):
    hwLabels = [];
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i];
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, k)
        #print("The classifier came back with %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    #print("\nThe total number of errors is: %d" % errorCount)
    #print("\nThe total error rate is: %f" % (errorCount/float(mTest)))
    return errorCount/float(mTest)

def main():
    errorrate = []
    m = 100
    fread = open("result.txt", 'w')
    fread.write("start@  " + ctime() + '\n')
    fread.write("K --------- errorrate\n")
    for k in range(m):
        errorrate.append(handwritingclasstest(k+1))
        fread.write(str(k+1))
        fread.write('         ')
        fread.write(str(errorrate[k]))
        fread.write('\n')
    fread.write("  end@  " + ctime() + '\n')
    fread.close()

if __name__ == '__main__':
    main()
