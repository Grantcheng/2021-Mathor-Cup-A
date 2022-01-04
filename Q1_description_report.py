import pandas as pd
from pandas_profiling import ProfileReport

# %%
dataset_path = 'data/Q1_train_set.xlsx'
dataset = pd.read_excel(dataset_path)
profile = ProfileReport(dataset, title='Q1: Train Set', plot={'dpi': 200, 'image_format': 'png'}, interactions=None)
profile = profile.to_html()
pd.to_pickle(profile, 'results/Q1_train_set_report.html')

# %%
dataset_path = 'data/Q1_test_set.xlsx'
dataset = pd.read_excel(dataset_path)
profile = ProfileReport(dataset, title='Q1: Test Set', plot={'dpi': 200, 'image_format': 'png'}, interactions=None)
profile = profile.to_html()
pd.to_pickle(profile, 'results/Q1_test_set_report.html')

# %%
dataset_path = 'raw/Q1_train_pre_processed_basic.pkl'
x, y = pd.read_pickle(dataset_path)
dataset = pd.concat([x, y], axis=1)
profile = ProfileReport(dataset, title='Q1: Train Set Pre-processed Basic',
                        plot={'dpi': 200, 'image_format': 'png'}, interactions=None,
                        )
profile = profile.to_html()
pd.to_pickle(profile, 'results/Q1_train_set_basic_report.html')
