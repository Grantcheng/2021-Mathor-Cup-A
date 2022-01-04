import pandas as pd
from sklearn.pipeline import Pipeline
from pre_processor import DropColumns

x, y = pd.read_pickle('raw/Q1_train_pre_processed_basic.pkl')
remove_correlation = Pipeline([
    ('Drop-1', DropColumns(cols=['brand', 'serial', 'carCode', 'maketype', 'anonymous_5', 'anonymous_6', 'anonymous_8',
                                 'anonymous_9', 'anonymous_10', 'anonymous_11', 'registryAge', 'carHeight',
                                 # correlated to dependent variables:
                                 # 'modelyear', 'licenseAge', 'carLength', 'carWidth',
                                 ])),
])
x = remove_correlation.fit_transform(x)
pd.to_pickle([x, y], 'raw/Q1_train_non_corr_22.pkl')
pd.to_pickle(remove_correlation, 'raw/Q1_mdl_correlation.pkl')
