# -*- coding: utf-8 -*-
from sklearn.linear_model import RandomizedLasso

# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RandomizedLasso.html#sklearn.linear_model.RandomizedLasso
estimators = []
index = 0
# for scaling in [x * 0.1 for x in xrange(1, 10)]:
for random_state in xrange(1, 100):
    params = {
        # 'scaling': scaling,
        'sample_fraction': 1,
        'n_jobs': -1,
        'random_state': random_state
    }
    name = "RandomizedLasso{0}\t".format(index)+"error:{0}\ttime:{1}\t"+",".join([str(k)+':'+str(v) for k,v in params.items()])
    estimators.append((name, RandomizedLasso(**params)))
    index += 1