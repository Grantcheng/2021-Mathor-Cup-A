import pandas as pd
import numpy as np

# %% Load data from spreadsheet.
# x = pd.read_excel('data/Q1_test_set.xlsx')
# pd.to_pickle(x, 'raw/Q1_test_set.pkl')

# %% Load data from binary.
x = pd.read_pickle('raw/Q1_test_set.pkl')
index = x['carId'].copy()

# %% Apply pre-processing workflow.
transformers = pd.read_pickle('raw/Q1_mdl_pre_processed_basic.pkl')
x_transformer, y_transformer = transformers['pre_processed_mdl_x'], transformers['pre_processed_mdl_y']
x = x_transformer.transform(x)
remove_correlation = pd.read_pickle('raw/Q1_mdl_correlation.pkl')
x = remove_correlation.transform(x)

# %% Use the model to predict.
optimizer = pd.read_pickle('raw/Q1_mdl_refit_stacking.pkl')
y_hat = optimizer.predict(x)
y_hat_df = pd.DataFrame(data=y_hat[:, np.newaxis], columns=['price'])
y_hat_raw_df = y_transformer.inverse_transform(y_hat_df)

# %% Export results
y_hat_raw_df['carId'] = index
y_hat_raw_df = y_hat_raw_df[['carId', 'price']]
y_hat_raw_df.to_csv('results/AMCB2100001Test1.txt', header=None, index=None, sep='\t', mode='w', float_format='%.2f')
