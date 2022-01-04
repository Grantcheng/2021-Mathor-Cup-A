import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# %% Load data.
with open('raw/Q1_train_non_corr_22.pkl', 'rb') as f:
    x, y = pickle.load(f)
# with open('raw/Q1_train_non_corr_18.pkl', 'rb') as f:
#     x, y = pickle.load(f)

# %% Make train and valid set.
x_train, x_valid, y_train, y_valid = train_test_split(
    x.values, y.values.ravel(), shuffle=True, train_size=0.8, random_state=502383)

# %% Train model
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
optimizer.fit(x_train, y_train, sample_weight=1-y_train)

# %% Export trained model.
with open('raw/Q1_mdl_stacking.pkl', 'wb') as f:
    pickle.dump(optimizer, f)

# %% Import trained model.
# with open('raw/Q1_mdl_stacking.pkl', 'rb') as f:
#     optimizer = pickle.load(f)

# %% Evaluate trained model.
y_valid_hat = optimizer.predict(x_valid)

with open('raw/Q1_mdl_pre_processed_basic.pkl', 'rb') as f:
    pre_processed_models = pickle.load(f)
y_transformer = pre_processed_models['pre_processed_mdl_y']

y_valid_df = pd.DataFrame(data=y_valid[:, np.newaxis], columns=['price'])
y_valid_raw_df = y_transformer.inverse_transform(y_valid_df)
y_valid_raw = y_valid_raw_df.values.ravel()

y_valid_hat_df = pd.DataFrame(data=y_valid_hat[:, np.newaxis], columns=['price'])
y_valid_hat_raw_df = y_transformer.inverse_transform(y_valid_hat_df)
y_valid_hat_raw = y_valid_hat_raw_df.values.ravel()

ape = np.abs(y_valid_hat_raw - y_valid_raw) / y_valid_raw
m_ape = np.mean(ape)
accuracy = np.sum(ape <= 0.05) / ape.shape[0]
score = 0.2 * (1 - m_ape) + 0.8 * accuracy

# %% Visualize errors.
fig = plt.figure()
plt.hist(ape, bins=21, range=[0, 1], edgecolor='k', alpha=0.75)
plt.legend([f'Score = {score.__format__(".3f")}'])
fig.savefig('results/Q1_err_stacking.svg')
plt.close(fig)
