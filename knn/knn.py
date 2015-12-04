#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import operator

# 定义python数据导入的通用函数模块
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

# Python实现的构造分类器
def classify0(inX, dataSet, labels, k):
    dataSetSize =dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),
            key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# 读取测试文件并生成相应的Numpy矩阵
def testfile2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()

    # File lines
    numberOfLines = len(arrayOLines)
    # Numpy Matrix
    returnMat = zeros((numberOfLines, 3))
    index = 0

    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        index += 1

    return returnMat

# 读取并解析学习文件数据并且保存到一个特定的Numpy矩阵中
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()

    # File lines
    numberOfLines = len(arrayOLines)
    # Numpy Matrix
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0

    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    return returnMat, classLabelVector

# 对数据进行归一化处理
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

# 绘制散点图，可以修改ax.scatter中参数的值来绘制三个不同值中两个值的二位散点图
def datingShow():
    datingDataMat, datingLabel = file2matrix('datingTestSet.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)

#'''
#    ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
#    ax.set_xlabel('percentages of time spent playing video games')
#    ax.set_ylabel('liters of ice cream consumed per year')
#    plt.show()
#'''

#'''
#    ax.scatter(datingDataMat[:,0], datingDataMat[:,1])
#    ax.set_xlabel('frequent flier miles earned per year')
#    ax.set_ylabel('percentages of time spent playing video games')
#    plt.show()
#'''

#'''
    ax.scatter(datingDataMat[:,0], datingDataMat[:,2])
    ax.set_xlabel('frequent flier miles earned per year')
    ax.set_ylabel('liters of ice cream consumed per year')
    plt.show()
#'''

# 抽取90%的样本数据来学习，然后测试剩下的10%的样本来检测错误率
def datingClassTest(k):
    hoRatio = 0.1
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat,ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    print("With k = %d" % int(k))
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],\
                datingLabels[numTestVecs:m],int(k))
        if (classifierResult != datingLabels[i]):
        	errorCount += 1.0
        	print("the classifier came back with: %d, the real answer is: %d"\
        		% (classifierResult, datingLabels[i]))

    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

# 对于数据进行预测
def classifyPerson(k):
    resultList = ['not at all', 'in small doses', 'in large doses']
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    testdatingMat = testfile2matrix('test_data.txt')
    for i in range(testdatingMat.shape[0]):
        result = classify0((testdatingMat[i,:]-\
            minVals)/ranges, normMat, datingLabels,k)
        print("You will probably like this person: ",\
            resultList[result - 1])

def main():

# 展示二维散点图
    datingShow()

# 执行python knn.py >> k-testresult可以看出k=4的时候的错误率最低
#    for k in range(1, 901):
#        datingClassTest(k)

# 待测试的数据个数为10个在test_data.txt文件中
    classifyPerson(4)

if __name__ == '__main__':
    main()
