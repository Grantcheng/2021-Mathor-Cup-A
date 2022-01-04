import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

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

# %% Train the model.
arima = ARIMA(price, order=[6, 1, 3])
result = arima.fit()
with open('results/Q3_arima_result.txt', 'w') as f:
    f.write(result.summary().__str__())
    f.write(f'\nMAE = {result.mae}\n')

# %% Periodicity analysis.
roots = result.arroots[np.imag(result.arroots) != 0]
T = 2 * np.pi / np.arccos(np.real(roots) / np.linalg.norm(roots))
T = pd.to_timedelta(T / periods * (end_date - start_date)) / pd.to_timedelta(1, unit='days')

# %% ARCH Test.
result_ljung = acorr_ljungbox(result.resid ** 2, lags=12)
