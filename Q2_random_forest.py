import pickle

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV

# %% Load data.
with open('raw/Q2_pre_processed.pkl', 'rb') as f:
    x, y = pickle.load(f)

# %% Train model
optimizer = BayesSearchCV(
    RandomForestRegressor(n_jobs=-1, random_state=502383),
    {
        'max_depth': (20, 50),  # integer valued parameter
        'max_leaf_nodes': (12000, 15000),  # integer valued parameter
    },
    n_iter=32, cv=5
)
optimizer.fit(x, y)

# %% Export trained model.
with open('raw/Q2_mdl_bayes_random_forest.pkl', 'wb') as f:
    pickle.dump(optimizer, f)

# %% Import trained model.
with open('raw/Q2_mdl_bayes_random_forest.pkl', 'rb') as f:
    optimizer = pickle.load(f)

# %% Model evaluation.
y_hat = optimizer.predict(x)
MAE = mean_absolute_error(y, y_hat)
print(MAE)

# %% Feature importance.
feature_importance = pd.DataFrame({
    'Name':  x.columns,
    'Feature importance': optimizer.best_estimator_.feature_importances_,
})
feature_importance = feature_importance.sort_values(by='Feature importance', axis=0, ascending=False)
feature_importance.to_excel('results/Q2_feature_importance.xlsx', index=False)
