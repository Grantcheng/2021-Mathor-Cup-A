import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import matplotlib.pyplot as plt

# %% Load data.
with open('raw/Q1_train_non_corr_22.pkl', 'rb') as f:
    x, y = pickle.load(f)
# with open('raw/Q1_train_non_corr_18.pkl', 'rb') as f:
#     x, y = pickle.load(f)

# %% Make train and valid set.
x_train, x_valid, y_train, y_valid = train_test_split(
    x.values, y.values.ravel(), shuffle=True, train_size=0.8, random_state=502383)

# %% Train model
optimizer = BayesSearchCV(
    XGBRegressor(n_estimators=100, verbosity=0, n_jobs=-1, random_state=502383),
    {
        'learning_rate': Real(0.0001, 1, prior='log-uniform'),
        'max_depth': Integer(10, 50),
        'reg_alpha': Real(0, 1, prior='uniform'),  # L1 regularization
        'reg_lambda': Real(0, 1, prior='uniform'),  # L2 regularization
    },
    n_iter=32, cv=5
)
optimizer.fit(x_train, y_train, sample_weight=1-y_train)

# %% Export trained model.
with open('raw/Q1_mdl_bayes_xgboost.pkl', 'wb') as f:
    pickle.dump(optimizer, f)

# %% Import trained model.
with open('raw/Q1_mdl_bayes_xgboost.pkl', 'rb') as f:
    optimizer = pickle.load(f)

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
fig.savefig('results/Q1_err_bayes_xgboost.svg')
plt.close(fig)
