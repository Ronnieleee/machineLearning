# 使用K-近邻值算法做手写识别

(1) 收集数据：默认提供在文件里面的数据来源

(2) 准备数据：使用Python解析文本文件knn中的img2vector 函数来实现

(3) 分析数据：

(4) 训练算法：无

(5) 测试算法：测试的数据来源见文件数据

(6) 使用算法：通过测试测试文件目录下的测试数据，根据K的值来获取最低错误率

## 文件
学习的数据来源于: [trainingDigits Directory](./trainingDigits/)

测试的数据来源于: [testDigits Directory](./testDigits/)

测试的结果存储于: [result.txt](./result.txt)

## 结果分析
当k取3的时候的预测错误率最低，识别度最高。