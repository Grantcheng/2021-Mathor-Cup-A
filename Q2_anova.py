import numpy as np
import pandas as pd
from scipy.stats import f

# %% Load data.
x, y = pd.read_pickle('raw/Q2_pre_processed.pkl')
feature_importance = pd.read_excel('results/Q2_feature_importance.xlsx')
transformer = pd.read_pickle('raw/Q2_mdl_pre_processed.pkl')

# %% Group the dependent variable.
y_bins = np.quantile(y, q=[0.2, 0.4, 0.6, 0.8])
y_grouped = np.sum(y.values[:, np.newaxis] < y_bins, axis=1)

# %%
features = feature_importance.loc[feature_importance['Feature importance'] >= 0.05, 'Name'].tolist()
anova = pd.DataFrame(columns=features)

totals = x[features].values
totals_std = np.std(totals, axis=0)
n = totals.shape[0]
for period in range(y_bins.shape[0] + 1):
    samples = totals[y_grouped == period, :]
    m = samples.shape[0]
    F = np.std(samples, axis=0) / totals_std
    p = f.cdf(F, m-1, n-1)
    anova.loc[period, features] = p

# %%
anova.to_excel('results/Q2_ANOVA.xlsx', index=False)
