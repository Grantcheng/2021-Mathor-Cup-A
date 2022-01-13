import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


class LifePeriod(TransformerMixin):
    def __init__(self, push_date, pull_date, new_name, unit='days'):
        self.push_date = push_date
        self.pull_date = pull_date
        self.new_name = new_name
        self.unit = unit
        self.period = 1

    def fit(self, x, *args, **kwargs):
        duration = (x[self.pull_date] - x[self.push_date]) / pd.to_timedelta(1, unit=self.unit)
        self.period = np.nanmean(duration)
        return self

    def transform(self, x, *args, **kwargs):
        x_ = x.copy()
        duration = (x[self.pull_date] - x[self.push_date]) / pd.to_timedelta(1, unit=self.unit)
        duration[duration.isna()] = self.period + 3 * np.nanstd(duration)
        duration = - np.log(duration / self.period)
        duration[duration >= 3] = 3
        duration[duration <= -3] = -3
        x_[self.new_name] = duration
        return x_


class Discount(TransformerMixin):
    def __init__(self, raw_price, discounted_price, new_name):
        self.raw_price = raw_price
        self.discounted_price = discounted_price
        self.new_name = new_name

    def fit(self, x, *args, **kwargs): return self

    def transform(self, x, *args, **kwargs):
        x_ = x.copy()
        x_[self.new_name] = x[self.discounted_price] / x[self.raw_price]
        return x_
