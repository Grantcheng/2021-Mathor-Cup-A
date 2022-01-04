import pandas as pd
from sklearn.base import TransformerMixin
import json

class LifePeriod(TransformerMixin):
    def __init__(self, push_date, pull_date, new_name, unit='days'):
        self.push_date = push_date
        self.pull_date = pull_date
        self.new_name = new_name
        self.unit = unit

    def fit(self, x, *args, **kwargs): return self

    def transform(self, x, *args, **kwargs):
        x_ = x.copy()
        x_.loc[x[self.pull_date].isna(), self.new_name] = 0
        duration = (x.loc[~x[self.pull_date].isna(), self.pull_date] -
                    x.loc[~x[self.pull_date].isna(), self.push_date]) / pd.to_timedelta(1, unit=self.unit)
        x_.loc[~x[self.pull_date].isna(), self.new_name] = 1 / (duration + 1)
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
