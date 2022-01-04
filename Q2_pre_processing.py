from sklearn.pipeline import Pipeline
from pre_processor import *
from pre_processor_extended import *

# %% Load data from spreadsheet.
# prices = pd.read_excel('data/Q2_train_set.xlsx')
# pd.to_pickle(prices, 'raw/Q2_train_set.pkl')

# %% Load data from binary.
car_features = pd.read_pickle('raw/Q1_train_set.pkl')
prices = pd.read_pickle('raw/Q2_train_set.pkl')

# %% Join tables.
dataset = pd.merge(left=car_features, right=prices, on='carId')

# %% Pre-processing pipeline.
transformer = Pipeline([
    # Feature construction
    ('TargetEncoder', TargetEncoder(cols=['brand', 'serial', 'model', 'seatings', 'country',
                                          'color', 'cityId', 'carCode', 'maketype', 'oiltype', 'anonymous_1',
                                          'anonymous_2', 'anonymous_3', 'anonymous_6', 'anonymous_8',
                                          'anonymous_9', 'anonymous_10', 'anonymous_11', 'anonymous_14'])),
    ('Duration-RegisterDate', TimeDuration(start_date='registerDate', end_date='tradeTime', new_name='registryAge')),
    ('Duration-LicenseDate', TimeDuration(start_date='licenseDate', end_date='tradeTime', new_name='licenseAge')),
    ('Duration-Anony7', TimeDuration(start_date='anonymous_7', end_date='tradeTime', new_name='anonymous_7_age')),
    ('Discount-based-on-push', Discount(raw_price='pushPrice', discounted_price='price', new_name='discount_push')),
    ('Discount-based-on-new', Discount(raw_price='newprice', discounted_price='price', new_name='discount_new')),
    ('Log', LogarithmicTransform(cols=['anonymous_4', 'anonymous_5'])),
    ('ModelYear', YearUntilNow(this_year='tradeTime', cols=['modelyear'])),
    ('Anony13', YearMonthUntilNow(this_month='tradeTime', cols=['anonymous_13'])),
    ('Volume', VolumeParser(new_names=['carLength', 'carWidth', 'carHeight'], col='anonymous_12')),
    ('Is-sold', LifePeriod(push_date='pushDate', pull_date='withdrawDate', new_name='InvSoldDays')),
    ('CleanDateTimeColumns', DropColumns(cols=['carId', 'tradeTime', 'registerDate', 'licenseDate', 'anonymous_7',
                                               'anonymous_15', 'pushDate', 'withdrawDate', 'pullDate',
                                               'pushPrice', 'updatePriceTimeJson', 'price', 'newprice'])),
    ('ForceFloat64', ForceFloat64()),
    # Fill missing values
    ('Fill-Zero', FillingZero(cols=['carCode', 'country', 'maketype', 'anonymous_1', 'anonymous_8', 'anonymous_9',
                                    'anonymous_10', 'anonymous_11', 'anonymous_14', 'model', 'cityId'])),
    ('Fill-Average', FillingAverage(cols=['modelyear', 'gearbox', 'anonymous_4', 'anonymous_5', 'anonymous_7_age',
                                          'carLength', 'carWidth', 'carHeight', 'anonymous_13'])),
    # Zipping 0~1
    ('Abnormal', GaussianAbnormal(cols=['mileage', 'transferCount', 'displacement', 'gearbox',
                                        'anonymous_4', 'anonymous_5', 'anonymous_7_age', 'registryAge',
                                        'licenseAge', 'carLength', 'carWidth', 'carHeight', 'anonymous_13',
                                        ])),
    ('Zipping', Zipping()),
    ('Drop-high-corr', DropColumns(cols=['brand', 'serial', 'carCode', 'maketype', 'anonymous_5', 'anonymous_6',
                                         'anonymous_8', 'anonymous_9', 'anonymous_10', 'anonymous_11', 'registryAge',
                                         'carHeight']))
])

# %% Export.
dataset = transformer.fit_transform(dataset)
y = dataset['InvSoldDays']
x = dataset.drop(['InvSoldDays'], axis=1)
pd.to_pickle([x, y], 'raw/Q2_pre_processed.pkl')
pd.to_pickle(transformer, 'raw/Q2_mdl_pre_processed.pkl')
