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

#knn.datingShowPL()
#knn.datingShowFL()
#knn.datingShowFP()

knn.datingClassTest(3);
knn.classifyPerson(4);
