# -*- coding: utf-8 -*-
"""
Created on Sat Jun 04 05:47:56 2016

@author: jiashua
"""

import sys
import os
import time
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn import metrics, cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RandomizedLasso
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

import math
from itertools import izip

SEED = 25
rng = np.random.RandomState(1)

def mape(ypred, ytrue):
    """ returns the mean absolute percentage error """
    idx = ytrue != 0.0
    return np.mean(np.abs(ypred[idx] - ytrue[idx]) / ytrue[idx])


def PostProcess(file, data):
    fa = open(file)
    fb = open(data)
    
    with open(file.replace('.txt', '_out.csv'), 'w') as fout:
        for x, y in izip(fa, fb):
            value = float(x.strip())
            value = math.floor(value)
            if value < 1:
                value = 1

            strs = y.split('\t')
            if len(strs) < 3:
                print '[error]len(strs) < 3', len(strs)
            date = strs[0] + '-' + strs[1]
            dist = strs[2]

            fout.write(dist+','+date+','+str(value)+'\n')

# path definition
BASE_DIR    = os.getcwd() # os.path.dirname(os.path.abspath(__file__))
train_path  = os.path.join(BASE_DIR, 'trainFillZero_fea.bin')
test_path   = os.path.join(BASE_DIR, 'testFillZero_fea.bin')
test_txt    = os.path.join(BASE_DIR, 'testFillZero.txt')

scoredir = os.path.join(BASE_DIR, 'output/{0}'.format(time.strftime('%Y%m%d-%H%M%S')))
if not os.path.exists(scoredir):
    os.makedirs(scoredir)
    
scorefmt = os.path.join(scoredir, 'score_{0}.txt')
log_filepath  = os.path.join(scoredir, 'score.log')

params = {'n_estimators': 10, 'max_depth': 4, 'min_samples_split': 1, 'learning_rate': 0.01, 'loss': 'ls'}

# estimators = [
    # ("LinearRegression", LinearRegression(normalize=True)),

    # 0.72
    # ("AdaBoost1", AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=5, random_state=rng)),
    # 0.66
    # ("AdaBoost2", AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=8, random_state=rng)),
    # 0.71
    # ("AdaBoost3", AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=8, random_state=rng)),
    # 0.70
    # ("AdaBoost4", AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=10, random_state=rng)),
    # 0.72
    # ("AdaBoost5", AdaBoostRegressor(DecisionTreeRegressor(max_depth=7), n_estimators=8, random_state=rng)),

    # 0.71
    # ("Bagging1", BaggingRegressor(DecisionTreeRegressor())),
    # 0.64
    # ("Bagging2", BaggingRegressor(DecisionTreeRegressor(max_depth=5))),
    # 0.68
    # ("Bagging3", BaggingRegressor(DecisionTreeRegressor(max_depth=7))),
    # 0.66
    # ("Bagging4", BaggingRegressor(DecisionTreeRegressor(max_depth=10))),

    # ("ExtraTree1", ExtraTreesRegressor(20)),
    # ("ExtraTree2", ExtraTreesRegressor(10)),
    # ("ExtraTree3", ExtraTreesRegressor(30)),

    # ("RandomizedLasso", RandomizedLasso(random_state=42))
    # ("GradientBoostRegression", GradientBoostingRegressor(**params)),

    # 0.75
    # ("RandomForest1",RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=5, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)),
    # 0.72
    # ("RandomForest2",RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=8, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)),
    # 0.71
    # ("RandomForest3",RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)),

    # ("SGDRegressor",SGDRegressor(penalty='elasticnet', alpha=0.01, l1_ratio=0.25, fit_intercept=True))
# ]

from estimator import estimators

if __name__ == "__main__":
    with open(log_filepath, 'w') as fout:
        print 'load file {0} ...'.format(train_path)
        fout.write('load file {0} ...\n'.format(train_path))

        # training_data = np.loadtxt(file, delimiter='\t')
        training_data = np.fromfile(train_path, dtype=np.float32)
        training_row = len(training_data) / 415
        training_data = training_data.reshape((training_row, 415))

        y = training_data[:,0]
        X = training_data[:,1:]

        test_data = np.fromfile(test_path, dtype=np.float32)
        test_row = len(test_data) / 414
        test_data = test_data.reshape((test_row, 414))

        print X.shape
        print y.shape

        X_tr, X_tt, y_tr, y_tt = cross_validation.train_test_split(X, y, test_size=0.33, random_state=42)

        idx = 0
        minerror = sys.maxint

        for n, (name, estimator) in enumerate(estimators):
            start_time = time.time()
            estimator.fit(X_tr, y_tr)
            error = mape(estimator.predict(X_tt), y_tt)
            elapsed_time = time.time() - start_time
            if error < minerror:
                minerror
                idx = n

            print name.format(round(error, 5), elapsed_time)
            fout.write(name.format(round(error, 5), elapsed_time)+'\n')

    # output
    estimator = estimators[idx][1]
    estimator.fit(X, y)
    testresult = estimator.predict(test_data)
    np.savetxt(scorefmt.format(idx), testresult, fmt='%.2f')

    PostProcess(scorefmt.format(idx), test_txt)
        
        
        
        

