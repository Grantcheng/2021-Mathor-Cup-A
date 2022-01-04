import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa import stattools

# %% Load data.
car_features = pd.read_pickle('raw/Q1_train_set.pkl')
prices = pd.read_pickle('raw/Q2_train_set.pkl')
dataset = pd.merge(left=car_features, right=prices, on='carId')
transaction = dataset.loc[~dataset['withdrawDate'].isna(), ['price', 'withdrawDate']]

# %% Build date slicer.
periods = 227
start_date = transaction['withdrawDate'].min()
end_date = transaction['withdrawDate'].max()
scale_date = np.linspace(start_date.value, end_date.value, periods + 1)
scale_date = pd.to_datetime(scale_date)
date_slicer = pd.cut(transaction['withdrawDate'].values, scale_date, ordered=True)
transaction['date_idx'] = date_slicer.codes.astype('int32')

# %% Make time series.
price = np.full(periods + 1, 0, dtype=np.float32)
for period, sub_transaction in transaction.groupby('date_idx'):
    price[period] = np.sum(sub_transaction['price'])

# %% ADF Test -> i.
param_i = dict()
for i in range(4):
    adf = stattools.adfuller(np.diff(price, n=i), autolag='t-stat')
    param_i[i] = adf[1]

# %% ACF -> q, PACF -> p
i = 1
n_lags = 40
acf = stattools.acf(np.diff(price, n=i), nlags=n_lags)
pacf = stattools.pacf(np.diff(price, n=i), nlags=n_lags)

fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
axes[0].stem(acf)
axes[0].plot([0, n_lags], [0.05, 0.05], color='k')
axes[0].plot([0, n_lags], [-0.05, -0.05], color='k')
axes[0].set_ylabel('ACF')
axes[0].set_xlabel('Lags')
axes[1].stem(pacf)
axes[1].plot([0, n_lags], [0.05, 0.05], color='k')
axes[1].plot([0, n_lags], [-0.05, -0.05], color='k')
axes[1].set_ylabel('PACF')
axes[1].set_xlabel('Lags')
fig.savefig(f'results/Q3_acf_pacf.svg')
plt.close(fig)
