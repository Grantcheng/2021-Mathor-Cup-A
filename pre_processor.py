import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


class TargetEncoder(TransformerMixin):
    def __init__(self, cols=None):
        """
        Perform the following action in each of the columns: Transfer each unique value to the frequency of itself
        among the training set.
        :param cols: the columns to transfer values
        """
        self.cols = cols
        self.frequency_dict = dict()

    def fit(self, x, *args, **kwargs):
        for col in self.cols:
            content, frequency = np.unique(x.loc[~x[col].isna(), col], return_counts=True)
            frequency = frequency / max(x.shape[0], 1)
            self.frequency_dict[col] = dict(zip(content, frequency))
        return self

    def transform(self, x, *args, **kwargs):
        x_ = x.copy()
        for col in self.cols:
            x_[col] = x_[col].apply(lambda val: self.frequency_dict[col].get(val), convert_dtype='float32')
        return x_


class AverageValueEncoder(TransformerMixin):
    def __init__(self, cols=None):
        """
        Perform the following action in each of the columns: Group the samples by unique values in a column,
        and transfer each unique value to the average value of corresponding dependent variable.
        :param cols: the columns to transfer values
        """
        self.cols = cols
        self.value_dict = dict()

    def fit(self, x, y, *args, **kwargs):
        for col in self.cols:
            self.value_dict[col] = {
                val: np.nanmean(y.loc[x[col] == val])
                for val in np.unique(x.loc[~x[col].isna(), col])
            }
        return self

    def transform(self, x, *args, **kwargs):
        x_ = x.copy()
        for col in self.cols:
            x_[col] = x_[col].apply(lambda val: self.value_dict[col].get(val), convert_dtype='float32')
        return x_


class DropColumns(TransformerMixin):
    def __init__(self, cols=None):
        """
        Drop columns
        :param cols: The columns to drop.
        """
        self.cols = cols

    def fit(self, x, *args, **kwargs): return self

    def transform(self, x, *args, **kwargs):
        return x.drop(columns=self.cols)


class TimeDuration(TransformerMixin):
    def __init__(self, start_date, end_date, new_name, unit='days'):
        """
        Create a new column, representing the duration of two columns.
        :param start_date: The column containing start date of the duration.
        :param end_date: The column containing end date of the duration.
        :param new_name: The new name of created column
        :param unit: The unit of duration, referring to
                     https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html
        """
        self.start_date = start_date
        self.end_date = end_date
        self.new_name = new_name
        self.unit = unit

    def fit(self, x, *args, **kwargs): return self

    def transform(self, x, *args, **kwargs):
        x_ = x.copy()
        x_[self.new_name] = (x[self.start_date] - x[self.end_date]) / pd.to_timedelta(1, unit=self.unit)
        return x_


class YearUntilNow(TransformerMixin):
    def __init__(self, this_year, cols):
        """
        Calculate how many years until now.
        :param this_year: The column represent this year.
        :param cols: The columns representing the year.
        """
        self.this_year = this_year
        self.cols = cols

    def fit(self, x, *args, **kwargs): return self

    def transform(self, x, *args, **kwargs):
        x_ = x.copy()
        x_[self.cols] = x[self.cols].apply(lambda val: x[self.this_year].dt.year - val)
        return x_


class YearMonthUntilNow(TransformerMixin):
    def __init__(self, this_month, cols):
        """
        Calculate how many month until now.
        :param this_month: The column representing this month.
        :param cols: The columns representing '%Y%m'.
        """
        self.this_month = this_month
        self.cols = cols

    def fit(self, x, *args, **kwargs): return self

    def transform(self, x, *args, **kwargs):
        x_ = x.copy()
        for col in self.cols:
            full_date_string = x[col].apply(lambda val: np.nan if np.isnan(val) else
                val.__int__().__str__() + '01').astype('str')
            full_date = pd.to_datetime(full_date_string, format='%Y%m%d')
            x_[col] = (x[self.this_month] - full_date) / pd.to_timedelta(1, unit='days')
        return x_


class LogarithmicTransform(TransformerMixin):
    def __init__(self, cols, padding=0):
        """
        Perform $x -> ln(x + padding)$.
        :param cols: The columns to perform logarithmic transformation.
        :param padding: Add a constant to the variable.
        """
        self.cols = cols
        self.padding = padding

    def fit(self, x, *args, **kwargs):
        for col in self.cols:
            assert x[col].min() + self.padding > 0, f'The minimum value of {col} should be non-positive.'
        return self

    def transform(self, x, *args, **kwargs):
        x_ = x.copy()
        x_[self.cols] = x[self.cols].apply(lambda val: np.log(val + self.padding))
        return x_

    def inverse_transform(self, x, *args, **kwargs):
        x_ = x.copy()
        x_[self.cols] = x[self.cols].apply(lambda val: np.exp(val) - self.padding)
        return x_


class VolumeParser(TransformerMixin):
    def __init__(self, new_names, col):
        """
        Parse the string representing volume to numeric.
        :param new_names: List[length name, width name, height name] The name of columns representing the 3
            components of volume.
        :param col: The column representing volume.
        """
        self.col = col
        self.length_name, self.width_name, self.height_name = new_names

    def fit(self, x, *args, **kwargs): return self

    def transform(self, x, *args, **kwargs):
        x_ = x.copy()
        x_[[self.length_name, self.width_name, self.height_name]] = x[self.col].str.split('*', expand=True)
        del x_[self.col]
        return x_


class FillingZero(TransformerMixin):
    def __init__(self, cols, zero=0):
        """
        Filling missing values as zero.
        :param cols:
        """
        self.cols = cols
        self.zero = zero

    def fit(self, x, *args, **kwargs): return self

    def transform(self, x, *args, **kwargs):
        x_ = x.copy()
        x_[self.cols] = x[self.cols].fillna(self.zero)
        return x_


class FillingAverage(TransformerMixin):
    def __init__(self, cols):
        """
        Filling missing values as the average of this variable.
        :param cols:
        """
        self.cols = cols
        self.average_dict = {}

    def fit(self, x, *args, **kwargs):
        average_all = np.nanmean(x[self.cols], axis=0)
        self.average_dict = dict(zip(self.cols, average_all))
        return self

    def transform(self, x, *args, **kwargs):
        x_ = x.copy()
        x_[self.cols] = x[self.cols].fillna(self.average_dict)
        return x_


class FillingAverageWithoutExtreme(FillingAverage):
    def __init__(self, cols, lower_pct=.05, upper_pct=.95):
        """
        Filling missing values as the average of this variable after deleting extreme values.
        :param cols:
        :param lower_pct: Values lower than the lower percentile are extreme values.
        :param upper_pct: Values higher than the upper percentile are extreme values.
        """
        super(FillingAverageWithoutExtreme, self).__init__(cols)
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, x, *args, **kwargs):
        n = x.shape[0]
        self.average_dict = {
            col: np.nanmean(np.sort(x[col])[round(self.lower_pct * n):round(self.upper_pct * n)])
            for col in self.cols
        }
        return self


class ForceFloat64(TransformerMixin):
    def __init__(self): pass

    def fit(self, x, *args, **kwargs): return self

    @staticmethod
    def transform(x, *args, **kwargs): return x.astype('float64')


class GaussianAbnormal(TransformerMixin):
    def __init__(self, cols, std_range=3, upper_only=False):
        """
        Move abnormal values into "mean ± std_range * std" assuming variables satisfying Gaussian distribution.
        :param cols: The columns to perform Gaussian abnormal processing.
        :param std_range: Values outside "mean ± std_range * std" are abnormal ones.
        """
        self.cols = cols
        self.std_range = std_range
        self.lower_bound = None
        self.upper_bound = None
        self.upper_only = upper_only

    def fit(self, x, *args, **kwargs):
        average = np.nanmean(x[self.cols], axis=0)
        stdev = np.nanstd(x[self.cols], axis=0)
        self.lower_bound = dict(zip(self.cols, average - self.std_range * stdev))
        self.upper_bound = dict(zip(self.cols, average + self.std_range * stdev))
        return self

    def transform(self, x, *args, **kwargs):
        x_ = x.copy()
        for col in self.cols:
            x_.loc[x[col] > self.upper_bound[col], col] = self.upper_bound[col]
            if not self.upper_only:
                x_.loc[x[col] < self.lower_bound[col], col] = self.lower_bound[col]
        return x_

    @staticmethod
    def inverse_transform(x, *args, **kwargs): return x


class Zipping(TransformerMixin):
    def __init__(self, cols=None, min_=0, max_=1):
        """
        This estimator scales and translates each feature individually such that it is in the given range on the
        training set, e.g. between zero and one.
        :param cols: The columns to perform zipping.
        :param min_: The lower bound of the zipped interval.
        :param max_: The upper bound of the zipped interval.
        """
        self.cols = cols
        self.min_ = min_
        self.max_ = max_
        self.lower_bound = None
        self.upper_bound = None

    def fit(self, x, *args, **kwargs):
        if self.cols is None:
            self.cols = x.columns
        self.lower_bound = np.nanmin(x[self.cols], axis=0)
        self.upper_bound = np.nanmax(x[self.cols], axis=0)
        return self

    def transform(self, x, *args, **kwargs):
        x_ = x.copy()
        x_[self.cols] = (x[self.cols] - self.lower_bound) / (self.upper_bound - self.lower_bound)
        x_[self.cols] = x_[self.cols] * (self.max_ - self.min_) + self.min_
        return x_

    def inverse_transform(self, x, *args, **kwargs):
        x_ = x.copy()
        x_[self.cols] = (x[self.cols] - self.min_) / (self.max_ - self.min_)
        x_[self.cols] = x_[self.cols] * (self.upper_bound - self.lower_bound) + self.lower_bound
        return x_
