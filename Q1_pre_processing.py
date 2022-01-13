from sklearn.pipeline import Pipeline
from pre_processor import *

# %% Load data from spreadsheet.
# dataset = pd.read_excel('data/Q1_train_set.xlsx')
# pd.to_pickle(dataset, 'raw/Q1_train_set.pkl')

# %% Load data from binary.
dataset = pd.read_pickle('raw/Q1_train_set.pkl')

# %% Separating independent and dependent variables.
y = dataset[['price']]
x = dataset.drop(['price'], axis=1)

# %% Define pre-processing pipeline for independent variables.
x_transformer = Pipeline([
    # Feature construction
    ('TargetEncoder-RelatedToBrand', TargetEncoder(cols=['brand', 'serial', 'model', 'seatings', 'country'])),
    ('TargetEncoder-Others', TargetEncoder(cols=['color', 'cityId', 'carCode', 'maketype', 'oiltype', 'anonymous_1',
                                                 'anonymous_2', 'anonymous_3', 'anonymous_6', 'anonymous_8',
                                                 'anonymous_9', 'anonymous_10', 'anonymous_11', 'anonymous_14'])),
    ('Duration-RegisterDate', TimeDuration(start_date='registerDate', end_date='tradeTime', new_name='registryAge')),
    ('Duration-LicenseDate', TimeDuration(start_date='licenseDate', end_date='tradeTime', new_name='licenseAge')),
    ('Duration-Anony7', TimeDuration(start_date='anonymous_7', end_date='tradeTime', new_name='anonymous_7_age')),
    ('Log', LogarithmicTransform(cols=['newprice', 'anonymous_4', 'anonymous_5'])),
    ('ModelYear', YearUntilNow(this_year='tradeTime', cols=['modelyear'])),
    ('Anony13', YearMonthUntilNow(this_month='tradeTime', cols=['anonymous_13'])),
    ('Volume', VolumeParser(new_names=['carLength', 'carWidth', 'carHeight'], col='anonymous_12')),
    ('CleanDateTimeColumns', DropColumns(cols=['carId', 'tradeTime', 'registerDate', 'licenseDate', 'anonymous_7',
                                               'anonymous_15'])),
    ('ForceFloat64', ForceFloat64()),
    # Fill missing values
    ('Fill-Zero', FillingZero(cols=['carCode', 'country', 'maketype', 'anonymous_1', 'anonymous_8', 'anonymous_9',
                                    'anonymous_10', 'anonymous_11', 'anonymous_14', 'model', 'cityId'])),
    ('Fill-Average', FillingAverage(cols=['modelyear', 'gearbox', 'anonymous_4', 'anonymous_5', 'anonymous_7_age',
                                          'carLength', 'carWidth', 'carHeight', 'anonymous_13'])),
    # Zipping 0~1
    ('Abnormal', GaussianAbnormal(cols=['mileage', 'transferCount', 'displacement', 'gearbox', 'newprice',
                                        'anonymous_4', 'anonymous_5', 'anonymous_7_age', 'registryAge',
                                        'licenseAge', 'carLength', 'carWidth', 'carHeight', 'anonymous_13',
                                        ])),
    ('Zipping', Zipping()),
])

# %% Define pre-processing pipeline for the dependent variable.
y_transformer = Pipeline([
    ('Log-Y', LogarithmicTransform(cols=['price'])),
    ('Abnormal', GaussianAbnormal(cols=['price'])),
    ('Zipping', Zipping()),
])

# %% Export pre-processing results.
x = x_transformer.fit_transform(x)
y = y_transformer.fit_transform(y)
pd.to_pickle([x, y], 'raw/Q1_train_pre_processed_basic.pkl')
pd.to_pickle(
    {'pre_processed_mdl_x': x_transformer, 'pre_processed_mdl_y': y_transformer},
    'raw/Q1_mdl_pre_processed_basic.pkl'
)
