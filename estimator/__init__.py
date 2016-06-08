# -*- coding: utf-8 -*-

# todo: bagging_regressor
# todo: randomized_lasso

# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import SGDRegressor
# from sklearn.ensemble import RandomForestRegressor


# params = {'n_estimators': 10, 'max_depth': 4, 'min_samples_split': 1, 'learning_rate': 0.01, 'loss': 'ls'}

# estimators = [
    # ("LinearRegression", LinearRegression(normalize=True)),

    # 0.71
    # ("RandomForest3",RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)),

    # ("SGDRegressor",SGDRegressor(penalty='elasticnet', alpha=0.01, l1_ratio=0.25, fit_intercept=True))
# ]



# params = {
# }
# name = "AdaBoost{0}\t".format(index)+"error:{0}\ttime:{1}\t"+",".join([str(k)+':'+str(v) for k,v in params.items()])
# estimators.append((name, AdaBoostRegressor(**params)))
# index += 1