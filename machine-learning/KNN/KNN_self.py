# @Time : 2020/6/12 10:04 上午 
# @Author : elroy
# @File : KNN_self.py 
# @Software: PyCharm
import numpy as np
from math import sqrt
from collections import Counter
from .metrics import accuracy_score


class KNNclassifier:

    def __init__(self,k):
        """初始化knn分类器"""
        assert k >=1 ,'k must be valid'
        self.k = k
        self.X_train = None
        self.y_train = None


    def fit(self,X_train,y_train):
        """根据训练数据集X_train和y_train训练knn分类器"""

        assert X_train.shape[0] == y_train.shape[0],\
        "the size of x_train must be euqal to the size of y_train"
        assert self.k <= X_train.shape[0],\
        "the size of X_train must be at least k"

        self._X_train = X_train
        self._y_train = y_train
        return self


    def predict(self,X_predict):
        """给定待测数据集x——predict 返回表示x_predict的结果向量"""
        assert self._X_train is not None and self._y_train is not None,\
        "must fit before predict"
        assert X_predict.shape[1] == self._X_train.shape[1],\
        "the feature number must equal"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self,x):
        """给定单个待预测数据x，返回x的预测结果值"""
        assert  x.shape[0] == self._X_train.shape[1],\
        "the feature number of x must be equal to x_train"

        distances = [sqrt(np.sum(x_train - x) **2 )
                     for x_train in self._X_train]

        nearest = np.argsort(distances)

        topk_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topk_y)

        return votes.most_common(1)[0][0]


    def score(self,X_test,y_test):
        """根据测试数据集x_test,y_test确定当前模型准确率"""

        y_predict = self.predict(X_test)
        return accuracy_score(y_test,y_predict)

    def __repr__(self):
        return "KNN(k=%d)" % self.k