import tensorflow as tf
from tensorflow.keras.layers import *
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# %% Load data.
with open('raw/Q1_train_non_corr_22.pkl', 'rb') as f:
    x, y = pickle.load(f)
# with open('raw/Q1_train_non_corr_18.pkl', 'rb') as f:
#     x, y = pickle.load(f)

# %% Make train and valid set.
x_train, x_valid, y_train, y_valid = train_test_split(
    x.values, y.values, shuffle=True, train_size=0.8, random_state=502383)

# %% Constant.
idx = 1  # model index
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64/'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# %% Train the model.
l0 = Input(shape=(x.shape[1], ))
l1 = Dense(64, activation='relu')(l0)
l2 = Dense(64, activation='relu')(l1)
l5 = Dense(64, activation='relu')(l2)
l3 = Dense(16, activation='relu')(l5)
l4 = Dense(1, activation='relu')(l3)

my_model = tf.keras.models.Model(l0, l4)
my_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss=tf.losses.Huber(delta=0.02))
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=60)
save_best = tf.keras.callbacks.ModelCheckpoint(f'raw/Q1_mdl_nn_{idx}.h5', monitor="val_loss", save_best_only=True)

train_log = my_model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=5000, batch_size=30000,
                         callbacks=[stop_early, save_best])
with open(f'raw/Q1_history_nn_{idx}.pkl', 'wb') as f:
    pickle.dump(train_log.history, f)

# %% Import trained model.
my_model = tf.keras.models.load_model(f'raw/Q1_mdl_nn_{idx}.h5')

# %% Evaluate trained model.
y_valid_hat = my_model.predict(x_valid)
with open('raw/Q1_mdl_pre_processed_basic.pkl', 'rb') as f:
    pre_processed_models = pickle.load(f)
y_transformer = pre_processed_models['pre_processed_mdl_y']

y_valid_df = pd.DataFrame(data=y_valid, columns=['price'])
y_valid_raw_df = y_transformer.inverse_transform(y_valid_df)
y_valid_raw = y_valid_raw_df.values

y_valid_hat_df = pd.DataFrame(data=y_valid_hat, columns=['price'])
y_valid_hat_raw_df = y_transformer.inverse_transform(y_valid_hat_df)
y_valid_hat_raw = y_valid_hat_raw_df.values

ape = np.abs(y_valid_hat_raw - y_valid_raw) / y_valid_raw
m_ape = np.mean(ape, axis=0)
accuracy = np.sum(ape <= 0.05, axis=0) / ape.shape[0]
score = 0.2 * (1 - m_ape) + 0.8 * accuracy

# %% Visualize errors.
fig = plt.figure()
plt.hist(ape, bins=21, range=[0, 1], edgecolor='k', alpha=0.75)
plt.legend([f'Score = {score.__float__().__format__(".3f")}'])
fig.savefig(f'results/Q1_err_nn_{idx}.svg')
plt.close(fig)
