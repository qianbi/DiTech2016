# -*- coding: utf-8 -*-



params = {'n_estimators': 10, 'max_depth': 4, 'min_samples_split': 1, 'learning_rate': 0.01, 'loss': 'ls'}

# estimators = [
    # ("LinearRegression", LinearRegression(normalize=True)),
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
