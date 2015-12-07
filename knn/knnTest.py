#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import knn

group, labels = knn.createDataSet()
print("group :\n", group)
print("labels :\n", labels)

classifyResult = knn.classify([0, 0], group, labels, 3)
print("classifyResult :\n", classifyResult)
print("type of classifyResult :\n", type(classifyResult))

datingDataMat, datingLabels = knn.file2matrix('datingTestSet.txt')
#print("datingDataMat :\n", datingDataMat)
#print("datingLabels :\n", datingLabels)

normMat, ranges, minVals = knn.autoNorm(datingDataMat)
#print("normMat :\n", normMat)
print("ranges :\n", ranges)
print("minVals : \n", minVals)

knn.datingShowPL()
knn.datingShowFL()
knn.datingShowFP()

knn.datingClassTest(3);
