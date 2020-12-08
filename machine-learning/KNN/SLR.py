# @Time : 2020/6/16 3:03 下午 
# @Author : elroy
# @File : SLR.py 
# @Software: PyCharm
import numpy as np
class SimpleLinearRegression2:

    def __init__(self):
        """初始化 simple linear regression 模型"""
        self.a_ = None
        self.b_ = None

    def fit(self, X_train, y_train):
        """根据训练数据集x_train y_train 来训练simple linear regression 模型"""
        assert X_train.ndim == 1, \
            "simple linear regression can only solve single feature training data"
        assert len(X_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_predict"

        X_mean = np.mean(X_train)
        y_mean = np.mean(y_train)

        num = (X_train - X_mean).dot(y_train - y_mean)
        d = (X_train - X_mean).dot(X_train - X_mean)

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * X_mean

        return self

    def predict(self, X_predict):
        """给定带预测数据集X_predict，返回x_predict的结果向量"""
        assert X_predict.ndim == 1, \
            "simple linear regression can only solve single feature training data"
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict"

        return np.array([self._predict(X) for X in X_predict])

    def _predict(self, X_single):
        """给定单个带预测数据x_single，返回预测值"""
        return self.a_ * X_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression2()"
