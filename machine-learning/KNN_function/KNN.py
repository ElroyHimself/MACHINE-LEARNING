# @Time : 2020/6/11 10:49 下午 
# @Author : elroy
# @File : KNN.py 
# @Software: PyCharm
import numpy as np
from math import sqrt
import matplotlib as mlb
from collections import Counter

def KNN_classify(k,X_train,y_train,x):

    assert 1 <=k <=X_train.shape[0],'k must be valid'
    assert X_train.shape[0] == y_train.shape[0],\
        'the size of x_train must equal to x_train'
    assert X_train.shape[1] == x.shape[0],\
    'the feature number of x must be qual to x_train'

    distances = []
    for x_train in X_train:
        d = sqrt(np.sum((x_train-x)**2))
        distances.append(d)

    nearest = np.argsort(distances)

    topk_y = [y_train[i] for  i in nearest[:k]]
    votes = Counter(topk_y)

    return votes.most_common(1)[0][0]