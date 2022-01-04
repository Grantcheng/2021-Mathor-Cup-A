import pickle

from sklearn.ensemble import StackingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

with open('raw/Q1_train_non_corr_22.pkl', 'rb') as f:
    x, y = pickle.load(f)
x, y = x.values, y.values.ravel()

optimizer = StackingRegressor(
    estimators=[
        ('Random forest', RandomForestRegressor(n_jobs=-1, random_state=502383, max_depth=49, max_leaf_nodes=11274)),
        ('Adaboost', AdaBoostRegressor(
            base_estimator=DecisionTreeRegressor(max_depth=50, criterion='mae', random_state=502383),
            n_estimators=50, learning_rate=0.1
        )),
        ('XGBoost', XGBRegressor(n_estimators=100, verbosity=0, n_jobs=-1, random_state=502383, learning_rate=0.11,
                                 max_depth=10, reg_alpha=0, reg_lambda=1)),
    ],
    final_estimator=LinearRegression(), n_jobs=-1
)
optimizer.fit(x, y, sample_weight=1-y)

with open('raw/Q1_mdl_refit_stacking.pkl', 'wb') as f:
    pickle.dump(optimizer, f)
