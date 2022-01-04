import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from bayes_opt import BayesianOptimization

# %% Load data.
x, _ = pd.read_pickle('raw/Q2_pre_processed.pkl')
feature_importance = pd.read_excel('results/Q2_feature_importance.xlsx')
features = feature_importance.loc[feature_importance['Feature importance'] >= 0.05, 'Name'].tolist()
x = x[features]

# %% Perform clustering.
def dbscan(eps):
    dbscan_ = DBSCAN(eps=eps, min_samples=100, n_jobs=-1)
    dbscan_.fit(x)
    if len(np.unique(dbscan_.labels_)) < 2:
        return -1
    return silhouette_score(x, dbscan_.labels_)

optimizer = BayesianOptimization(
    f=dbscan, pbounds={'eps': (0, 1)}, random_state=48934
)
optimizer.maximize(init_points=16, n_iter=16)
print('Best silhouette score:', optimizer.max['target'])
dbscan__ = DBSCAN(eps=optimizer.max['params']['eps'], min_samples=100, n_jobs=-1)
dbscan__.fit(x)
class_labels = dbscan__.labels_

# %% Export the model.
pd.to_pickle(dbscan__, 'raw/Q3_mdl_dbscan.pkl')

# %% Import the model.
dbscan__ = pd.read_pickle('raw/Q3_mdl_dbscan.pkl')

# %%
class_unique = np.unique(class_labels)
average = pd.DataFrame(columns=features, index=class_unique)
class_count = []
for label in class_unique:
    sub_x = x.loc[class_labels == label, :].values
    average.loc[label, :] = (sub_x.mean(axis=0) - x.mean(axis=0)) / x.std(axis=0)
    class_count.append(sub_x.shape[0])
average['Count'] = class_count
average['Class label'] = class_unique
average.to_excel('results/Q3_clustered_average.xlsx', index=False)
print(average)
