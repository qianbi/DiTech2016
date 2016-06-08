# -*- coding: utf-8 -*-
from sklearn.ensemble import GradientBoostingRegressor


# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor




estimators = []
index = 0
for loss in ['ls', 'lad', 'huber', 'quantile']:
    params = {
        'loss': loss
    }
    name = "GradientBoosting{0}\t".format(index)+"error:{0}\ttime:{1}\t"+",".join([str(k)+':'+str(v) for k,v in params.items()])
    estimators.append((name, GradientBoostingRegressor(**params)))
    index += 1