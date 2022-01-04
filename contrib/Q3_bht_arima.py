import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from contrib.BHT_ARIMA import BHTARIMA

# %% Load data.
x, y = pd.read_pickle('raw/Q3_pre_processed.pkl')
std_ = pd.read_pickle('raw/Q3_standard_scaler.pkl')

# %% Train the model and predict.
bht = BHTARIMA(p=5, d=1, q=2, taus=[x.shape[0], 51], Rs=[51, 51], K=100, tol=0.01, verbose=0, Us_mode=4)
y_hat = bht.fit_transform(x)

# %% Evaluate the result.
error = np.abs(y_hat - y)
ljung = acorr_ljungbox(error, lags=12)
print(ljung)
score = mean_absolute_error(y, y_hat)
print("MAE:", score)
