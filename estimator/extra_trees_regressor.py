# -*- coding: utf-8 -*-
from sklearn.ensemble import ExtraTreesRegressor

estimators = []
index = 0
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html#sklearn.ensemble.ExtraTreesRegressor
for n_estimators in xrange(5, 7):
    for max_features in ['auto', None]:
        for max_depth in xrange(7, 10):
            for min_samples_split in xrange(2, 5):
                for min_samples_leaf in xrange(1, 5):
                    for min_weight_fraction_leaf in [x * 0.1 for x in xrange(0, 5)]:
                        params = {
                            'n_estimators': n_estimators,
                            'max_features': max_features,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            'min_weight_fraction_leaf': min_weight_fraction_leaf
                        }
                        name = "ExtraTree{0}\t".format(index)+"error:{0}\ttime:{1}\t"+",".join([str(k)+':'+str(v) for k,v in params.items()])
                        estimators.append((name, ExtraTreesRegressor(**params)))
                        index += 1