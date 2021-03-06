# -*- coding: utf-8 -*-
from sklearn.tree import DecisionTreeRegressor

estimators = []
index = 0
for max_depth in [5]:
    for max_feature in ['auto', None]:
        for min_samples_split in xrange(2, 11):
            for min_samples_leaf in xrange(2, 11):
                for min_weight_fraction_leaf in [x * 0.01 for x in xrange(10)]:
                    for max_leaf_nodes in [None]:
                        for random_state in [None]:
                            params = {
                                'criterion': "mse",
                                'splitter': "best", # random best
                                'max_depth': max_depth, # None int
                                'max_features': max_feature, # None int float 'auto' 'sqrt' 'log2'
                                'min_samples_split': min_samples_split, # int2
                                'min_samples_leaf': min_samples_leaf, # int1
                                'min_weight_fraction_leaf': min_weight_fraction_leaf, # float [0, 0.5]
                                'max_leaf_nodes': max_leaf_nodes, # int or None
                                'random_state': random_state, # int or None
                            }
                            name = "DecisionTree{0}\t".format(index)+"error:{0}\ttime:{1}\t"+",".join([str(k)+':'+str(v) for k,v in params.items()])
                            estimators.append((name, DecisionTreeRegressor(**params)))
                            index += 1