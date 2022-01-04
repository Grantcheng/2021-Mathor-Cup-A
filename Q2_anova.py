import numpy as np
import pandas as pd

# %% Load data.
x, y = pd.read_pickle('raw/Q2_pre_processed.pkl')
feature_importance = pd.read_excel('results/Q2_feature_importance.xlsx')

# %% Group the dependent variable.
days_description = {
    1: 'Within a day', 7: 'Within a week', 14: 'Within two weeks', 30: 'Within a month', 91: 'Within a quarterly',
    182: 'Within half a year', 365: 'More than half a year'
}
y_days = (1 / y - 1).values
y_grouped = np.piecewise(y_days,
                         condlist=[y_days <= 1, (y_days > 1) & (y_days <= 7), (y_days > 7) & (y_days <= 14),
                                   (y_days > 14) & (y_days <= 30), (y_days > 30) & (y_days <= 91),
                                   (y_days > 91) & (y_days <= 182)],
                         funclist=list(days_description.keys())).astype('int32')

# %%
features = feature_importance.loc[feature_importance['Feature importance'] >= 0.05, 'Name'].tolist()
anova = pd.DataFrame(columns=features, index=list(days_description.keys()))
for period in list(days_description.keys()):
    anova.loc[period, features] = x[features].values[y_grouped == period, :].std(axis=0) / x[features].std(axis=0)

# %%
anova['Description'] = list(days_description.values())
anova.to_excel('results/Q2_ANOVA.xlsx', index=False)
