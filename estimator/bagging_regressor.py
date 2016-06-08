# -*- coding: utf-8 -*-
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

estimators = []
index = 0
    # todo
    # params = {
    #     DecisionTreeRegressor(max_depth=5,min_samples_leaf=3)
    # }
    # name = "Bagging{0}\t".format(index)+"error:{0}\ttime:{1}\t"+",".join([str(k)+':'+str(v) for k,v in params.items()])
    # estimators.append((name, BaggingRegressor(**params)))
    # index += 1