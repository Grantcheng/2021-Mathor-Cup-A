import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

# %% Standardization.
std_ = StandardScaler()
price = std_.fit_transform(price[:, np.newaxis]).ravel()
pd.to_pickle(std_, 'raw/Q3_standard_scaler.pkl')

# %% Make moving window.
def moving_window(ts, k):
    """
    Make moving window samples from time series.
    :param ts: Time series.
    :param k: Length of the window.
    :return: x_, y_: fraction used as input, fraction used as output.
    """
    l = ts.shape[0]
    y_ = ts[k:]
    indices = np.tile(np.arange(k), [l-k, 1]) + np.arange(l-k)[:, np.newaxis]
    x_ = ts[indices]
    return x_, y_


x, y = moving_window(price, k=60)
pd.to_pickle([x, y], 'raw/Q3_pre_processed.pkl')
