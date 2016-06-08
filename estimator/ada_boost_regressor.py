# -*- coding: utf-8 -*-
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

import numpy as np
rng = np.random.RandomState(1)

estimators = []
index = 0
for n_estimators in xrange(48, 53):
    for loss in ['linear', 'square', 'exponential']:
        for learning_rate in xrange(1, 2):
            params = {
                'base_estimator': DecisionTreeRegressor(max_depth=5,min_samples_leaf=3),
                'n_estimators': n_estimators, # default: 50
                'learning_rate': learning_rate, # default: 1
                'loss': loss,  # linear square exponential
                'random_state': rng
            }

            name = "AdaBoost{0}\t".format(index)+"error:{0}\ttime:{1}\t"+",".join([str(k)+':'+str(v) for k,v in params.items()])
            estimators.append((name, AdaBoostRegressor(**params)))
            index += 1